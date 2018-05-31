import os
import cv2
import random
import cfg
import time
import numpy as np
import tensorflow as tf
from model.caffe2tf import load_model

def get_now_filepath():
    time_now = time.localtime()
    return "_".join(map(str, [time_now.tm_mon, time_now.tm_mday, time_now.tm_hour, time_now.tm_min]))

def load_pretrained_model(net_name, datapath, sess, ignore_missing):
    print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    print ">>>> loading pretrained model in %s"%(net_name)
    print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    assert os.path.exists(datapath)
    params_data = load_model(datapath)
        
    with tf.variable_scope(net_name, reuse=True):
        for key in params_data:
            with tf.variable_scope(key, reuse=True):
                for subkey in params_data[key]:
                    try:
                        var = tf.get_variable(subkey)
                        assert var.get_shape().as_list() == list(params_data[key][subkey].shape)
                        sess.run(var.assign(params_data[key][subkey]))
                        print 'assign pretrain model ' + subkey + ' to ' + key
                    except ValueError:
                        print 'ignore ' + key
                        if not ignore_missing:
                            raise NotImplementedError

def upscale(x, scale):
    _, h, w, _  = x.shape
    assert h==w
    return tf.image.resize_images(x, (h*scale, w*scale), method=1)
                                                
def crop(image, resized_size, cropped_size):
    # image is of arbitrary size.
    # return a Tensor representing image of size cropped_size x cropped_size
    image = tf.image.resize_images(image, [resized_size, resized_size], method=tf.image.ResizeMethod.AREA)
    offset = tf.cast(tf.floor(tf.random_uniform([2], 0, resized_size - cropped_size + 1)), dtype=tf.int32)
    image = tf.image.crop_to_bounding_box(image, offset[0], offset[1], cropped_size, cropped_size)
    return image

def subtract_mean(image):
    # image is a Tensor.
    # return a Tensor.
    image = tf.cast(image, dtype=tf.float32)
    return image - tf.convert_to_tensor(cfg.PIXEL_MEANS, dtype=tf.float32)

def prep(image):
    # change range from [0, 256) to [-1, 1]
    # image is a Tensor.
    # return a float32 Tensor.
    image = tf.cast(image, dtype=tf.float32)
    return (image / 255.0) * 2 - 1


def invprep(image):
    # change range from [-1, 1] to [0, 256)
    # image is a float32 Tensor.
    # return a uint8 Tensor.
    image = (image + 1) / 2.0 * 255.9
    return image

def bgr2rgb(image):
    image = tf.cast(image, dtype=tf.uint8)
    return image[:,:,:,::-1]

def hwc2chw(image):
    if image.ndim == 4:
        # batch size
        return image.transpose(0, 3, 1, 2)
    elif image.ndim == 3:
        return image.transpose(2, 0, 1)
    else:
        raise NotImplementedError

def chw2hwc(image):
    if image.ndim == 4:
        # batch size
        return image.transpose(0, 2, 3, 1)
    elif image.ndim == 3:
        return image.transpose(1, 2, 0)
    else:
        raise NotImplementedError

def make_var(name, shape, initializer=None, trainable=True, regularizer=None):
    return tf.get_variable(name, shape, initializer=initializer, trainable=trainable, regularizer=regularizer)

def l2_regularizer(weight_decay=0.0005, scope=None):
    def regularizer(tensor):
        with tf.name_scope(scope, default_name='l2_regularizer', values=[tensor]):
            l2_weight = tf.convert_to_tensor(weight_decay, dtype=tensor.dtype.base_dtype, name='weight_decay')
            return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')
    return regularizer

def instance_norm(input, name="instance_norm"):
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        if name=='debug':
            return scale, offset, mean, variance, inv, normalized, scale*normalized + offset
        else:
            return scale*normalized + offset

def batch_norm(input, scope='batchnorm'):
    with tf.variable_scope(scope):
        input = tf.identity(input)
        dims = input.get_shape()
        if len(dims) == 4:
            channels = dims[3]
            offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
            scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
            mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        elif len(dims) == 2:
            channels = dims[1]
            offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
            scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
            mean, variance = tf.nn.moments(input, axes=[0], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized

def leaky_relu(input, alpha=0.3, name='leaky_relu'):
    return tf.maximum(alpha*input, input, name)

def relu(input):
    return tf.nn.relu(input)

def elu(input):
    return tf.nn.elu(input)

def validate_padding(padding):
    assert padding in ('SAME', 'VALID')

def conv(input, k_h, k_w, c_o, s_h, s_w, name, stddev=0.02, biased=True, group=1, bn=False, init='msra', pad='SAME', trainable=True):
    """
    k_h, k_w: height and width of kernel size
    c_o: channel of output
    s_h, s_w: height and width of convolution stride
    """

    # Verify that the padding is acceptable
    validate_padding(pad)
    
    # Get the number of channels in the input
    c_i = input.get_shape()[-1] # channel_input
    # Verify that the grouping parameter is valid
    assert c_i % group == 0
    assert c_o % group == 0
    # Convolution for a given input and kernel
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=pad)
    with tf.variable_scope(name) as scope:
        if init == 'msra':
            init_weights = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
        elif init == 'random':
            init_weights = tf.random_normal_initializer(stddev=stddev)
        else:
            raise Exception('Invalid init')
        kernel = make_var('weights', [k_h, k_w, c_i/group, c_o], init_weights, trainable, regularizer=l2_regularizer(cfg.WEIGHT_DECAY))

        if group == 1:
            h = convolve(input, kernel)
        else:
            input_groups = tf.split(input, group, 3)
            kernel_groups = tf.split(kernel, group, 3)

            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            h = tf.concat(output_groups, 3)

        if biased:
            init_bias = tf.constant_initializer(0.0)
            bias = make_var('biases', [c_o], init_bias, trainable)
            h = tf.nn.bias_add(h, bias)
        if bn:
            h = batch_norm(h)
        return h

        # if activation == 'relu':
        #     h = tf.nn.relu(h)
        # elif activation == 'leaky_relu':
        #     h = leaky_relu(h)
        # elif activation == 'tanh':
        #     h = tf.nn.tanh(h)
        # elif activation == 'elu':
        #     h = tf.nn.elu(h)
        # elif activation == None:
        #     h = h
        # return h

def upconv(input, c_o, ksize, stride, name, stddev=0.02, biased=False, bn=False, init='msra', pad='SAME', trainable=True):
    c_i = input.get_shape()[-1] # channel_input
    in_shape = tf.shape(input)
    if pad == 'SAME':
        output_shape = [in_shape[0], in_shape[1]*stride, in_shape[2]*stride, c_o]
    else:
        raise Exception('Sorry not support padding VALID')
    kernel_shape = [ksize, ksize, c_o, c_i]
    with tf.variable_scope(name) as scope:
        if init == 'msra':
            init_weights = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
        elif init == 'random':
            init_weights = tf.truncated_normal_initializer(stddev=stddev)
        else:
            raise Exception('Invalid init')
        kernel = make_var('weights', kernel_shape, init_weights, trainable, regularizer=l2_regularizer(cfg.WEIGHT_DECAY))
        h = tf.nn.conv2d_transpose(input, kernel, output_shape, [1, stride, stride, 1], padding=pad)
        h = tf.reshape(h, output_shape) # reshape is necessary
        if biased:
            init_bias = tf.constant_initializer(0.0)
            bias = make_var('biases', [c_o], init_bias, trainable)
            h = tf.nn.bias_add(h, bias)
        if bn:
            h = batch_norm(h)
        return h

        # if activation == 'relu':
        #     h = tf.nn.relu(h)
        # elif activation == 'leaky_relu':
        #     h = leaky_relu(h)
        # else:
        #     h = h
        # return h

def lrn(input, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(input,
                                             depth_radius=radius,
                                             alpha=alpha,
                                             beta=beta,
                                             bias=bias,
                                             name=name)

def max_pool(input, k_h, k_w, s_h, s_w, name, pad='SAME'):
    validate_padding(pad)
    return tf.nn.max_pool(input, 
                          ksize=[1, k_h, k_w, 1], 
                          strides=[1, s_h, s_w, 1], 
                          padding=pad, 
                          name=name)

def avg_pool(input, k_h, k_w, s_h, s_w, name, pad='SAME'):
    validate_padding(pad)
    return tf.nn.avg_pool(input, 
                          ksize=[1, k_h, k_w, 1], 
                          strides=[1, s_h, s_w, 1], 
                          padding=pad, 
                          name=name)

def fc(input, c_o, name, biased=True, bn=False, init='msra', trainable=True):
    with tf.variable_scope(name) as scope:
        if init == 'msra':
            init_weights = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
        else:
            raise Exception('Invalid init')
    
        input_shape = input.get_shape()
        if input_shape.ndims == 4:
            # The input is spatial. Vectorize it first
            dim = 1
            for d in input_shape[1:].as_list():
                dim *= d
            # Suit for caffe-style
            input = tf.reshape(tf.transpose(input, perm=[0, 3, 1, 2]), [-1, dim])
        
        c_i = input.get_shape()[-1]
        weights = make_var('weights', [c_i, c_o], init_weights, trainable, regularizer=l2_regularizer(cfg.WEIGHT_DECAY))
        
        h = tf.matmul(input, weights)
        if biased:
            init_bias = tf.constant_initializer(0.0)
            bias = make_var('biases', [c_o], init_bias, trainable)
            h = tf.nn.bias_add(h, bias)
        if bn:
            h = batch_norm(h)
        return h

        # if activation == 'relu':
        #     h = tf.nn.relu(h)
        # elif activation == 'leaky_relu':
        #     h = leaky_relu(h)
        # elif activation == 'tanh':
        #     h = tf.nn.tanh(h)
        # elif activation == 'elu':
        #     h = tf.nn.elu(h)
        # return h


def fcT(input, c_o, name, biased=True, bn=False, init='msra', trainable=True):
    with tf.variable_scope(name) as scope:
        if init == 'msra':
            init_weights = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
        else:
            raise Exception('Invalid init')
    
        input_shape = input.get_shape()
        if input_shape.ndims == 4:
            # The input is spatial. Vectorize it first
            dim = 1
            for d in input_shape[1:].as_list():
                dim *= d
            # Suit for caffe-style
            input = tf.reshape(tf.transpose(input, perm=[0, 3, 1, 2]), [-1, dim])
        
        c_i = input.get_shape()[-1]
        weights = make_var('weights', [c_o, c_i], init_weights, trainable, regularizer=l2_regularizer(cfg.WEIGHT_DECAY))
        h = tf.matmul(input, tf.transpose(weights))
        if biased:
            init_bias = tf.constant_initializer(0.0)
            bias = make_var('biases', [c_o], init_bias, trainable)
            h = tf.nn.bias_add(h, bias)
        if bn:
            h = batch_norm(h)
        return h
        
        # if activation == 'relu':
        #     h = tf.nn.relu(h)
        # elif activation == 'leaky_relu':
        #     h = leaky_relu(h)
        # elif activation == 'tanh':
        #     h = tf.nn.tanh(h)
        # elif activation == 'elu':
        #     h = tf.nn.elu(h)
        # return h


def sum_act(h, sparsity=False):
    tf.summary.histogram('activation/'+h.name, h)
    if sparsity:
        tf.summary.scalar('sparsity/'+h.name, tf.nn.zero_fraction(h))
