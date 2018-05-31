import os
import tensorflow as tf
from utils.util import conv, upconv, max_pool, lrn, fc, instance_norm, leaky_relu, relu

def generator_unet(image, map_feature, gf_dim, reuse=False, dropout=False, keep_prob=1.0, name="default"):

    assert len(map_feature.get_shape().as_list()) == 3
    map_feature = tf.nn.l2_normalize(map_feature, dim=2, name='normed_map_feature')
    with tf.variable_scope("generator_unet_%s"%(name), reuse=reuse) as vs:
        # image is 256 x 256 x input_c_dim
        # if reuse:
        #     tf.get_variable_scope().reuse_variables()
        # else:
        #     assert tf.get_variable_scope().reuse is False
        assert image.get_shape().as_list()[1:] == [256, 256, 3]

        e1 = instance_norm(conv(image, 4, 4, gf_dim, 2, 2, name='g_e1_conv', init='random', biased=False), 'g_bn_e1')
        e1 = tf.concat([tf.nn.l2_normalize(e1, dim=3, name='l2_normalized_e1'), tf.tile(map_feature[:, :, None, :], [1, 128, 128, 1])], axis=3)

        # test_e1 = instance_norm(e1)
        # e1 is (128 x 128 x self.gf_dim)
        e2 = instance_norm(conv(leaky_relu(e1, alpha=0.2), 4, 4, gf_dim*2, 2, 2, name='g_e2_conv', init='random', biased=False), 'g_bn_e2')
        # e2 is (64 x 64 x self.gf_dim*2)
        # e2 = tf.concat([tf.nn.l2_normalize(e2, dim=3, name='l2_normalized_e2'), tf.tile(map_feature[:, :, None, :], [1, 64, 64, 1])], axis=3)

        e3 = instance_norm(conv(leaky_relu(e2, alpha=0.2), 4, 4, gf_dim*4, 2, 2, name='g_e3_conv', init='random', biased=False), 'g_bn_e3')
        # e3 is (32 x 32 x self.gf_dim*4)
        e4 = instance_norm(conv(leaky_relu(e3, alpha=0.2), 4, 4, gf_dim*8, 2, 2, name='g_e4_conv', init='random', biased=False), 'g_bn_e4')
        # e4 is (16 x 16 x self.gf_dim*8)
        e5 = instance_norm(conv(leaky_relu(e4, alpha=0.2), 4, 4, gf_dim*8, 2, 2, name='g_e5_conv', init='random', biased=False), 'g_bn_e5')
        # e5 is (8 x 8 x self.gf_dim*8)
        e6 = instance_norm(conv(leaky_relu(e5, alpha=0.2), 4, 4, gf_dim*8, 2, 2, name='g_e6_conv', init='random', biased=False), 'g_bn_e6')
        # e6 is (4 x 4 x self.gf_dim*8)
        e7 = instance_norm(conv(leaky_relu(e6, alpha=0.2), 4, 4, gf_dim*8, 2, 2, name='g_e7_conv', init='random', biased=False), 'g_bn_e7')
        # e7 is (2 x 2 x self.gf_dim*8)
        # e8 = instance_norm(conv(leaky_relu(e7, alpha=0.2), 4, 4, gf_dim*8, 2, 2, name='g_e8_conv', init='random', activation=None, biased=False), 'g_bn_e8')

        # remove instance norm
        e8 = conv(leaky_relu(e7, alpha=0.2), 4, 4, gf_dim*8, 2, 2, name='g_e8_conv', init='random', biased=False)
        # e8 is (1 x 1 x self.gf_dim*8)

        # self.upper_norm_image_feat = tf.nn.l2_normalize(self.upper_image_feat, dim=1, name='upper_norm_image_feat')
        # e8 = tf.nn.l2_normalize(e8, dim=3, name='l2_normalized_e8')
        # d1 = upconv(tf.nn.relu(tf.concat([e8, map_feature[:, :, None, :]], axis=3)), gf_dim*8, 4, 2, name='g_d1', init='random', activation=None)

        d1 = upconv(relu(e8), gf_dim*8, 4, 2, name='g_d1')
        if dropout:
            d1 = tf.nn.dropout(d1, keep_prob)
        d1 = tf.concat([instance_norm(d1, 'g_bn_d1'), e7], 3)
        # d1 is (2 x 2 x self.gf         _dim*8*2)

        d2 = upconv(relu(d1), gf_dim*8, 4, 2, name='g_d2', init='random')
        if dropout:
            d2 = tf.nn.dropout(d2, keep_prob)
        d2 = tf.concat([instance_norm(d2, 'g_bn_d2'), e6], 3)
        # d2 is (4 x 4 x self.gf_dim*8*2)

        d3 = upconv(relu(d2), gf_dim*8, 4, 2, name='g_d3', init='random')
        if dropout:
            d3 = tf.nn.dropout(d3, keep_prob)
        d3 = tf.concat([instance_norm(d3, 'g_bn_d3'), e5], 3)
        # d3 is (8 x 8 x self.gf_dim*8*2)

        d4 = upconv(relu(d3), gf_dim*8, 4, 2, name='g_d4', init='random')
        d4 = tf.concat([instance_norm(d4, 'g_bn_d4'), e4], 3)
        # d4 is (16 x 16 x self.gf_dim*8*2)

        d5 = upconv(relu(d4), gf_dim*4, 4, 2, name='g_d5', init='random')
        d5 = tf.concat([instance_norm(d5, 'g_bn_d5'), e3], 3)

        d6 = upconv(d5, gf_dim*2, 4, 2, name='g_d6', init='random')
        d6 = tf.concat([instance_norm(d6, 'g_bn_d6'), e2], 3)
        # d6 is (64 x 64 x self.gf_dim*2*2)

        d7 = upconv(relu(d6), gf_dim, 4, 2, name='g_d7', init='random')
        d7 = tf.concat([instance_norm(d7, 'g_bn_d7'), e1], 3) # one to multi
        # d7 is (128 x 128 x self.gf_dim*1*2)

        d8 = upconv(relu(d7), 3, 4, 2, name='g_d8', init='random')
        # d8 is (256 x 256 x output_c_dim)

        variables = tf.contrib.framework.get_variables(vs)

    return tf.nn.tanh(d8), variables

def generator_caffenet_fc6(input_feat, reuse=False, trainable=False):
    with tf.variable_scope('generator', reuse=reuse) as vs:
        assert input_feat.get_shape().as_list()[-1] == 4096
        # input_feat = tf.placeholder(tf.float32, shape=(None, 4096), name='feat')

        relu_defc7 = leaky_relu(fc(input_feat, 4096, name='defc7', trainable=trainable))
        relu_defc6 = leaky_relu(fc(relu_defc7, 4096, name='defc6', trainable=trainable))
        relu_defc5 = leaky_relu(fc(relu_defc6, 4096, name='defc5', trainable=trainable))
        reshaped_defc5 = tf.reshape(relu_defc5, [-1, 256, 4, 4])
        relu_deconv5 = leaky_relu(upconv(tf.transpose(reshaped_defc5, perm=[0, 2, 3, 1]), 256, 4, 2, 
                              'deconv5', biased=True, trainable=trainable))
        relu_conv5_1 = leaky_relu(upconv(relu_deconv5, 512, 3, 1, 'conv5_1', biased=True, trainable=trainable))
        relu_deconv4 = leaky_relu(upconv(relu_conv5_1, 256, 4, 2, 'deconv4', biased=True, trainable=trainable))
        relu_conv4_1 = leaky_relu(upconv(relu_deconv4, 256, 3, 1, 'conv4_1', biased=True, trainable=trainable))
        relu_deconv3 = leaky_relu(upconv(relu_conv4_1, 128, 4, 2, 'deconv3', biased=True, trainable=trainable))
        relu_conv3_1 = leaky_relu(upconv(relu_deconv3, 128, 3, 1, 'conv3_1', biased=True, trainable=trainable))
        deconv2 = leaky_relu(upconv(relu_conv3_1, 64, 4, 2, 'deconv2', biased=True, trainable=trainable))
        deconv1 = leaky_relu(upconv(deconv2, 32, 4, 2, 'deconv1', biased=True, trainable=trainable))
        deconv0 = upconv(deconv1, 3, 4, 2, 'deconv0', biased=True, trainable=trainable)

    variables = tf.contrib.framework.get_variables(vs)

    return deconv0, variables