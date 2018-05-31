import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

from resnet_config import Config

import datetime
import numpy as np
import os
import time

MOMENTUM = 0.9
MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_DECAY = 0.00004
FC_WEIGHT_STDDEV = 0.01
RESNET_VARIABLES = 'resnet_variables'
UPDATE_OPS_COLLECTION = 'resnet_update_ops'
IMAGENET_MEAN_BGR = [103.062623801, 115.902882574, 123.151630838, ]

activation = tf.nn.relu

def encoder_resnet(images, layer=101, name='encoder', reuse=False, is_training=False):

    if layer == 50:
        num_blocks = [3, 4, 6, 3]
    elif layer == 101:
        num_blocks = [3, 4, 23, 3]
    elif layer == 152:
        num_blocks = [3, 8, 36, 3]
    else:
        raise NotImplementedError

    c = Config()
    c['is_training'] = tf.convert_to_tensor(is_training, dtype='bool', name='is_training')
    c['bottleneck']  = True
    c['ksize'] = 3
    c['stride'] = 1
    c['use_bias'] = False
    c['num_blocks'] = num_blocks
    c['stack_stride'] = 2

    with tf.variable_scope('encoder_resnet', reuse=reuse) as vs:
        x = images
        with tf.variable_scope('scale1'):
            c['conv_filters_out'] = 64
            c['ksize'] = 7
            c['stride'] = 2
            x = conv(x, c)
            x = bn(x, c)
            x = activation(x)

        with tf.variable_scope('scale2'):
            x = _max_pool(x, ksize=3, stride=2)
            c['num_blocks'] = num_blocks[0]
            c['stack_stride'] = 1
            c['block_filters_internal'] = 64
            x = stack('scale2', x, c)

        with tf.variable_scope('scale3'):
            c['num_blocks'] = num_blocks[1]
            c['block_filters_internal'] = 128
            assert c['stack_stride'] == 2
            x = stack('scale3',x, c)

        with tf.variable_scope('scale4'):
            c['num_blocks'] = num_blocks[2]
            c['block_filters_internal'] = 256
            x = stack('scale4', x, c)

        with tf.variable_scope('scale5'):
            c['num_blocks'] = num_blocks[3]
            c['block_filters_internal'] = 512
            x = stack('scale5', x, c)

        # post-net
        x = tf.reduce_mean(x, axis=[1, 2], name="avg_pool")
    variables = tf.contrib.framework.get_variables(vs)

    return x, variables

def stack(name, x, c):
    for n in range(c['num_blocks']):
        s = c['stack_stride'] if n == 0 else 1
        c['block_stride'] = s
        with tf.variable_scope('block%d' % (n + 1)):
            x = block(name, x, c)
    # if self.is_summary:
    #   for i in range(min(x.get_shape()[-1],5)):
    #       tf.summary.image(name+'/ch'+str(i), tf.expand_dims(x[:,:,:,i], -1), max_outputs=4)
    return x

def block(name, x, c):
    filters_in = x.get_shape()[-1]
    m = 4 if c['bottleneck'] else 1
    filters_out = m * c['block_filters_internal']

    shortcut = x

    c['conv_filters_out'] = c['block_filters_internal']

    if c['bottleneck']:
        with tf.variable_scope('a'):
            c['ksize'] = 1
            c['stride'] = c['block_stride']
            x = conv(x, c)
            x = bn(x, c)
            x = activation(x)

        with tf.variable_scope('b'):
            x = conv(x, c)
            x = bn(x, c)
            x = activation(x)

        with tf.variable_scope('c'):
            c['conv_filters_out'] = filters_out
            c['ksize'] = 1
            assert c['stride'] == 1
            x = conv(x, c)
            x = bn(x, c)
    else:
        with tf.variable_scope('A'):
            c['stride'] = c['block_stride']
            assert c['ksize'] == 3
            x = conv(x, c)
            x = bn(x, c)
            x = activation(x)

        with tf.variable_scope('B'):
            c['conv_filters_out'] = filters_out
            assert c['ksize'] == 3
            assert c['stride'] == 1
            x = conv(x, c)
            x = bn(x, c)

    with tf.variable_scope('shortcut'):
        if filters_out != filters_in or c['block_stride'] != 1:
            c['ksize'] = 1
            c['stride'] = c['block_stride']
            c['conv_filters_out'] = filters_out
            shortcut = conv(shortcut, c)
            shortcut = bn(shortcut, c)

    return activation(x + shortcut)

def bn(x, c):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    if c['use_bias']:
        bias = _get_variable('bias', params_shape,
                            initializer=tf.zeros_initializer(),
                            trainable=False)
        return x + bias

    axis = list(range(len(x_shape) - 1))

    beta = _get_variable('beta',
                        params_shape,
                        initializer=tf.zeros_initializer(),
                        trainable=False)
    gamma = _get_variable('gamma',
                        params_shape,
                        initializer=tf.ones_initializer(),
                        trainable=False)

    moving_mean = _get_variable('moving_mean',
                        params_shape,
                        initializer=tf.zeros_initializer(),
                        trainable=False)
    moving_variance = _get_variable('moving_variance',
                        params_shape,
                        initializer=tf.ones_initializer(),
                        trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                            mean, BN_DECAY,
                                                            zero_debias=False)
    update_moving_variance = moving_averages.assign_moving_average(
            moving_variance, variance, BN_DECAY, zero_debias=False)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)


    mean, variance = control_flow_ops.cond(
        c['is_training'], lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)

    return x

def conv(x, c):
    ksize = c['ksize']
    stride = c['stride']
    filters_out = c['conv_filters_out']

    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    weights = _get_variable('weights',
                            shape=shape,
                            dtype='float',
                            initializer=initializer,
                            weight_decay=CONV_WEIGHT_DECAY,
                            trainable=False)
    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')

def _max_pool(x, ksize=3, stride=2):
    return tf.nn.max_pool(x,ksize=[1, ksize, ksize, 1],
                        strides=[1, stride, stride, 1],
                        padding='SAME')


def _get_variable(name,
                shape,
                initializer,
                weight_decay=0.0,
                dtype='float',
                trainable=True):
    "A little wrapper around tf.get_variable to do weight decay and add to"
    "resnet collection"
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.GLOBAL_VARIABLES, RESNET_VARIABLES]
    return tf.get_variable(name,
                        shape=shape,
                        initializer=initializer,
                        dtype=dtype,
                        regularizer=regularizer,
                        collections=collections,
                        trainable=trainable)

