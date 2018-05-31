import tensorflow as tf
from utils.util import conv, max_pool, lrn, fc, relu

def encoder_caffenet(input_image, feat='fc6', reuse=False, trainable=False):
    with tf.variable_scope('encoder_caffenet', reuse=reuse) as vs:
        assert input_image.get_shape().as_list()[1:] == [227, 227, 3]
        # input_image = tf.placeholder(tf.float32, shape=(None, 227, 227, 3), name='input_image')
        relu1 = relu(conv(input_image, 11, 11, 96, 4, 4, pad='VALID', name='conv1', trainable=trainable))
        pool1 = max_pool(relu1, 3, 3, 2, 2, pad='VALID', name='pool1')
        norm1 = lrn(pool1, 2, 2e-05, 0.75, name='norm1')
        relu2 = relu(conv(norm1, 5, 5, 256, 1, 1, group=2, name='conv2', trainable=trainable))
        pool2 = max_pool(relu2, 3, 3, 2, 2, pad='VALID', name='pool2')
        norm2 = lrn(pool2, 2, 2e-05, 0.75, name='norm2')
        relu3 = relu(conv(norm2, 3, 3, 384, 1, 1, name='conv3', trainable=trainable))
        relu4 = relu(conv(relu3, 3, 3, 384, 1, 1, group=2, name='conv4', trainable=trainable))
        relu5 = relu(conv(relu4, 3, 3, 256, 1, 1, group=2, name='conv5', trainable=trainable))
        pool5 = max_pool(relu5, 3, 3, 2, 2, pad='VALID', name='pool5')
        fc6 = relu(fc(pool5, 4096, name='fc6', trainable=trainable))
        fc7 = relu(fc(fc6, 4096, name='fc7', trainable=trainable))
    variables = tf.contrib.framework.get_variables(vs)

    if feat == 'fc6':
        return fc6, variables
    elif feat == 'fc7':
        return fc7, variables
    else:
        raise NotImplementedError