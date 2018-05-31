import os
import cfg
import tensorflow as tf
from model.caffe2tf import save_model

FLAGS = tf.app.flags.FLAGS

def get_pretrain_encoder(net, load_type='np'):
    assert load_type in ['np', 'tf']
    if net == 'caffenet':
        if load_type == 'np':
            encoder_path = os.path.join(cfg.PRETRAIN_MODEL, 'caffenet', 'caffenet_params.npz')
            if not os.path.exists(encoder_path):
                encoder_deploy = os.path.join(cfg.PRETRAIN_MODEL, 'caffenet', 'caffenet.prototxt')
                encoder_model = os.path.join(cfg.PRETRAIN_MODEL, 'caffenet', 'caffenet.caffemodel')
                save_model(encoder_deploy, encoder_model, encoder_path)
        elif load_type == 'tf':
            encoder_path = os.path.join(cfg.PRETRAIN_MODEL, 'caffenet', 'caffenet_params')
        return encoder_path

    elif net == 'resnet':
        assert load_type == 'tf'
        encoder_path = os.path.join(cfg.PRETRAIN_MODEL, 'resnet', 'ResNet-L%s.ckpt'%(FLAGS.resnet_layer))
        return encoder_path

def get_pretrain_generator(net, load_type='np'):
    assert load_type in ['np', 'tf']
    if net == 'caffenet':
        generator_dir = os.path.join(cfg.GENERATOR_MODEL, 'caffenet')
        if load_type == 'np':
            generator_path = os.path.join(generator_dir, FLAGS.feat, 'caffenet_params.npz')
            if not os.path.exists(generator_path):
                generator_deploy = os.path.join(generator_dir, FLAGS.feat, 'generator.prototxt')
                generator_model  = os.path.join(generator_dir, FLAGS.feat, 'generator.caffemodel')
                save_model(generator_deploy, generator_model, generator_path)
        elif load_type == 'tf':
            generator_path = os.path.join(generator_dir, FLAGS.feat, 'caffenet_params')
    return generator_path

def get_pretrain_comparator(net, load_type='np'):
    if net == 'caffenet':
        return get_pretrain_encoder('caffenet', load_type=load_type)

def get_pretrain_classifier(net, load_type='np'):
    assert load_type in ['np', 'tf']
    if load_type == 'np':
        classification_path = os.path.join(cfg.CLASSIFICATION_MODEL, FLAGS.dataset, FLAGS.classifier, FLAGS.classifier_pretrain_model)
        if os.path.exists(classification_path):
            return classification_path
        else:
            raise NotImplementedError, 'You should pretrain a classifier early'
    elif load_type == 'tf':
        classification_path = os.path.join(cfg.CLASSIFICATION_MODEL, FLAGS.dataset, FLAGS.classifier, 'classifier')
        return classification_path    
