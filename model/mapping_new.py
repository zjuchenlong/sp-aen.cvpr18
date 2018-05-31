# -*- coding: utf-8 -*-

from utils.util import fc, fcT, leaky_relu
import tensorflow as tf

def visual2semantic_2layer(visual_feat, visual_size, semantic_size, name='default', reuse=False, dropout=False, keep_prob=1.0, activation='leaky_relu'):
    assert activation == 'leaky_relu'
    with tf.variable_scope('visual2semantic_2layer_%s'%(name), reuse=reuse) as vs:
        assert visual_feat.get_shape().as_list()[-1] == visual_size
        if dropout:
            visual_feat = tf.nn.dropout(visual_feat, keep_prob, name='visual_feat_dropout')
        map1 = leaky_relu(fc(visual_feat, 1024, biased=True, name='map1'))
        if dropout:
            map1 = tf.nn.dropout(map1, keep_prob, name='map1_dropout')
        semantic_feat = leaky_relu(fc(map1, semantic_size, biased=True, name='semantic_feat'))

    variables = tf.contrib.framework.get_variables(vs)
    return semantic_feat, variables

def semantic2visual_2d(semantic_feat, semantic_size, visual_size, name='default', reuse=False, dropout=False, keep_prob=1.0, activation='leaky_relu'):
    assert activation == 'leaky_relu'
    with tf.variable_scope('semantic2visual_2d_%s'%(name), reuse=reuse) as vs:
        assert semantic_feat.get_shape().as_list()[-2:] == [semantic_size, semantic_size]
        reshape_semantic_feat = tf.reshape(semantic_feat, [-1, semantic_size*semantic_size], name='reshape_semantic_feat')
        if dropout:
            reshape_semantic_feat = tf.nn.dropout(reshape_semantic_feat, keep_prob, name='reshape_semantic_feat_dropout')
        map2 = leaky_relu(fc(reshape_semantic_feat, semantic_size, biased=True, name='map2'))
        if dropout:
            map2 = tf.nn.dropout(map2, keep_prob, name='map2_dropout')
        visual_feat = leaky_relu(fc(map2, visual_size, biased=True, name='map1'))
    variables = tf.contrib.framework.get_variables(vs)
    return visual_feat, variables    

def semantic2visual_2layer(semantic_feat, semantic_size, visual_size, name='default', reuse=False, dropout=False, keep_prob=1.0, activation='leaky_relu'):
    assert activation == 'leaky_relu'
    with tf.variable_scope('semantic2visual_2layer_%s'%(name), reuse=reuse) as vs:
        assert semantic_feat.get_shape().as_list()[-1] == semantic_size
        if dropout:
            semantic_feat = tf.nn.dropout(semantic_feat, keep_prob, name='semantic_feat_dropout')
        map2 = leaky_relu(fc(semantic_feat, 1024, biased=True, name='map2'))
        if dropout:
            map2 = tf.nn.dropout(map2, keep_prob, name='map2_dropout')
        visual_feat = leaky_relu(fc(map2, visual_size, biased=True, name='map1'))
    variables = tf.contrib.framework.get_variables(vs)
    return visual_feat, variables

def semantic2semantic_2layer(semantic_feat, semantic_size, name='default', reuse=False, dropout=False, keep_prob=1.0, activation='leaky_relu'):
    assert activation == 'leaky_relu'
    with tf.variable_scope('semantic2semantic_2layer_%s'%(name), reuse=reuse) as vs:
        # assert semantic_feat.get_shape().as_list()[-1] == semantic_size
        if dropout:
            semantic_feat = tf.nn.dropout(semantic_feat, keep_prob, name='semantic_feat_dropout')
        map1 = leaky_relu(fc(semantic_feat, semantic_size, biased=True, name='map1'))
        if dropout:
            map1 = tf.nn.dropout(map1, keep_prob, name='map1_dropout')
        ret_semantic_feat = leaky_relu(fc(map1, semantic_size, biased=True, name='ret_semantic_feat'))

    variables = tf.contrib.framework.get_variables(vs)
    return ret_semantic_feat, variables