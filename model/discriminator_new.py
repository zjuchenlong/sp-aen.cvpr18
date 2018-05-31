import numpy as np
import tensorflow as tf
from utils.util import conv, fc, upscale, avg_pool, leaky_relu, instance_norm, relu
slim = tf.contrib.slim

def discriminator_patch(image, map_feature, df_dim, reuse=False, dropout=False, keep_prob=1.0, name="default"):
    assert len(map_feature.get_shape().as_list()) == 3
    map_feature = tf.nn.l2_normalize(map_feature, dim=2, name='normed_map_feature')
    with tf.variable_scope("discriminator_patch_%s"%(name), reuse=reuse) as vs:

        h0 = leaky_relu(instance_norm(conv(image, 4, 4, df_dim, 2, 2, name='d_h0_conv', init='random', biased=False)), alpha=0.2)
        h0 = tf.concat([tf.nn.l2_normalize(h0, dim=3, name='l2_normalized_h0'), tf.tile(map_feature[:, :, None, :], [1, 128, 128, 1])], axis=3)

        # h0 is (128 x 128 x self.df_dim)
        h1 = leaky_relu(instance_norm(conv(h0, 4, 4, df_dim*2, 2, 2, name='d_h1_conv', init='random', biased=False),'d_bn1'), alpha=0.2)
        # h1 is (64 x 64 x self.df_dim*2)
        h2 = leaky_relu(instance_norm(conv(h1, 4, 4, df_dim*4, 2, 2, name='d_h2_conv', init='random', biased=False), 'd_bn2'), alpha=0.2)
        # h2 is (32x 32 x self.df_dim*4)
        h3 = leaky_relu(instance_norm(conv(h2, 4, 4, df_dim*8, 1, 1, name='d_h3_conv', init='random', biased=False), 'd_bn3'), alpha=0.2)
        # h3 is (32 x 32 x self.df_dim*8)
        h4 = relu(conv(h3, 4, 4, 1, 1, 1, name='d_h3_pred'))
        # h4 is (32 x 32 x 1)
        variables = tf.contrib.framework.get_variables(vs)
    return h4, variables

def discriminator_image(input_image, real_input_label, fake_input_label, num_train_classes, name='image', reuse=False):
    with tf.variable_scope('discriminator_%s'%(name), reuse=reuse) as vs:
        dconv1 = relu(conv(input_image, 7, 7, 32, 4, 4, pad='VALID', name='dconv1'))
        dconv2 = relu(conv(dconv1, 5, 5, 64, 1, 1, pad='VALID', name='dconv2'))
        dconv3 = relu(conv(dconv2, 3, 3, 128, 2, 2, pad='VALID', name='dconv3'))
        dconv4 = relu(conv(dconv3, 3, 3, 256, 1, 1, pad='VALID', name='dconv4'))
        dconv5 = relu(conv(dconv4, 3, 3, 256, 2, 2, pad='VALID', name='dconv5'))

        dpool5 = avg_pool(dconv5, 11, 11, 11, 11, name='dpool5')
        dpool5_reshape = tf.reshape(dpool5, [-1, 256], name='dpool5_reshape')

        # label information
        real_label_feat = tf.one_hot(real_input_label, depth=num_train_classes)
        fake_label_feat = tf.one_hot(fake_input_label, depth=num_train_classes)
        label_feat = tf.concat([real_label_feat, fake_label_feat], axis=0)

        # SAVE GPU MEMORY
        Ffc1 = relu(fc(label_feat, 128, name='Ffc1'))
        Ffc2 = relu(fc(Ffc1, 128, name='Ffc2'))

        concat5 = tf.concat([dpool5_reshape, Ffc2], axis=1, name='concat5')
        drop5 = tf.nn.dropout(concat5, keep_prob=0.5, name='drop5')
        # SAVE GPU MEMORY
        dfc6 = relu(fc(drop5, 256, name='dfc6'))
        dfc7 = fc(dfc6, 1, name='dfc7')

    variables = tf.contrib.framework.get_variables(vs)

    return dfc7, variables

# def discriminator_cycle(vector, name='vector', reuse=False):
#     with tf.variable_scope('discriminator_%s'%(name), reuse=reuse) as vs:
#         output = fc(vector, 1, name='output', biased=True, trainable=True)
#     variables = tf.contrib.framework.get_variables(vs)
#     return output, variables

def discriminator_cycle(vector, name='vector', reuse=False):
    semantic_size = vector.get_shape().as_list()[-1]
    with tf.variable_scope('discriminator_%s'%(name), reuse=reuse) as vs:
        fc1 = leaky_relu(fc(vector, semantic_size, name='fc1', biased=True, trainable=False))
        output = fc(fc1, 1, name='output', biased=True, trainable=False)
    variables = tf.contrib.framework.get_variables(vs)
    return output, variables

# def discriminator_vector(real_vector, fake_vector, name='vector', reuse=False):
#     with tf.variable_scope('discriminator_%s'%(name), reuse=reuse) as vs:
#         input_  = tf.concat([real_vector, fake_vector], axis=0)
#         output = fc(input_, 1, name='output', biased=True, trainable=True)
#     variables = tf.contrib.framework.get_variables(vs)
#     return output, variables

