import os
import cfg
import numpy as np
import tensorflow as tf
from data.preprocessing import save_data
from sp_aen import SP_AEN

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('mode', '', 'trainf/traine/test/visualization the model')
tf.app.flags.DEFINE_string('dataset', '', 'cub|apy|awa2|sun, which dataset to do experiment')
# default setting
tf.app.flags.DEFINE_boolean('dropout', True, 'dropout')
tf.app.flags.DEFINE_float('keep_prob', 0.5, 'default for dropout')
tf.app.flags.DEFINE_string('encoder', 'resnet', 'caffenet|resnet, types of CNN network for encoder')
tf.app.flags.DEFINE_string('recon_encoder', 'caffenet', 'recon encoder')
tf.app.flags.DEFINE_integer('resnet_layer', 101, 'layer for resnet encoder')
tf.app.flags.DEFINE_string('generator', 'caffenet', 'types of CNN network for generator')
tf.app.flags.DEFINE_string('comparator', 'caffenet', 'types of CNN network for comparator')
# tf.app.flags.DEFINE_string('classifier', 'caffenet', 'types of CNN network for classifier')
tf.app.flags.DEFINE_string('feat', 'fc6', 'feature for reconstruct')
tf.app.flags.DEFINE_string('semantic', 'class_attr', 'type of semantic feature')
# tf.app.flags.DEFINE_string('classifier_pretrain_model', 'classifier.npz', 'pretrained classifer parameters')
tf.app.flags.DEFINE_string('rank_loss_type', 'random', 'mean|max|random')
tf.app.flags.DEFINE_float('margin', 0.2, 'margin for rank loss')
tf.app.flags.DEFINE_boolean('debug', True, 'debug mode')
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size for train set')
tf.app.flags.DEFINE_integer('test_batch_size', 1, 'batch size for test set')
# Opitmizer Adam
tf.app.flags.DEFINE_float('lr', 0.0001, 'learning rate')
tf.app.flags.DEFINE_float('beta1', 0.5, 'beta1 for adam optimizer')
tf.app.flags.DEFINE_float('beta2', 0.999, 'beta2 for adam optimizer')
# WGAN-GP
tf.app.flags.DEFINE_integer('n_critics', 5, 'number of n_critics')
tf.app.flags.DEFINE_float('wgan_lambda', 10, 'alpha for gradient penalty')
# File path
tf.app.flags.DEFINE_string('summary', './summary', 'directory to save summary')
tf.app.flags.DEFINE_string('summary_path', '', 'if given summary_path')
tf.app.flags.DEFINE_string('cachefile', './cache', 'store the cache result')
tf.app.flags.DEFINE_string('checkpoint', './checkpoint', 'path to store checkpoint')
tf.app.flags.DEFINE_string('train_checkpoint', '', 'model to reload for train')
tf.app.flags.DEFINE_string('test_checkpoint', '', 'model to reload for test')
tf.app.flags.DEFINE_string('visual_checkpoint', '', 'model to reload for visualization')
tf.app.flags.DEFINE_string('sae_checkpoint', '', 'model to reload for sae')
tf.app.flags.DEFINE_boolean('retrain_model', False, 'retrain model')
# Save
tf.app.flags.DEFINE_integer('ckpt_interval', 8000, 'save checkpoint at every time step')
tf.app.flags.DEFINE_integer('summary_interval', 100, 'show the summary at every time step')
tf.app.flags.DEFINE_integer('validate_interval', 50,  'evaluate the validation set')
tf.app.flags.DEFINE_boolean('save_checkpoint', True, 'whether to save the checkpoint')
tf.app.flags.DEFINE_string('model_name', 'SP-AEN', 'save model name')
tf.app.flags.DEFINE_integer('saveimage_interval', 100, 'the interval to save the reconstruct image')

# weights of different loss
tf.app.flags.DEFINE_float('alpha_rank', 1000, 'loss weight for classification') #
tf.app.flags.DEFINE_float('alpha_map_rank', 2, 'weight for map rank')
tf.app.flags.DEFINE_float('alpha_consist', 10, 'loss weight for constitence loss')
tf.app.flags.DEFINE_float('alpha_rank_dis', 50, '')
tf.app.flags.DEFINE_float('alpha_rank_gen', 50, '')
tf.app.flags.DEFINE_float('alpha_rank_regularization', 5e-6, 'rank regularization coeff')
tf.app.flags.DEFINE_float('alpha_gan_regularization', 5e-6, 'recon regularization coeff')
tf.app.flags.DEFINE_float('alpha_recon_img', 3e-2, 'loss weight for upper path in image pixel space')
tf.app.flags.DEFINE_float('alpha_recon_feat', 3, 'loss weight for upper path in image feat space')
# tf.app.flags.DEFINE_float('alpha_recon_img', 2e-5, 'loss weight for upper path in image pixel space')
# tf.app.flags.DEFINE_float('alpha_recon_feat', 2e-3, 'loss weight for upper path in image feat space')
# tf.app.flags.DEFINE_float('alpha_class', 2e-4, 'loss weight for classification')

# trainf mode
tf.app.flags.DEFINE_integer('training_F_step', 200000, 'epochs for training set')
tf.app.flags.DEFINE_integer('map_rank_norm', 1, 'target norm for train F')

# traine mode
tf.app.flags.DEFINE_integer('training_E_step', 200000, 'epochs for training set')
tf.app.flags.DEFINE_integer('decay_curriculum', 100000, 'Strong GAN loss for certeain period at the begining')


# For mode visualization
tf.app.flags.DEFINE_string('visual_path', 'vis', 'visualize the reconstruction result')
tf.app.flags.DEFINE_string('visual_name', '', 'recon image path name for reconstruction')

# For mode sae
tf.app.flags.DEFINE_string('sae_name', '', 'name for sae model')

# for class attr
tf.app.flags.DEFINE_boolean('negative_class_attr', False, 'make negative number for class attributes')
tf.app.flags.DEFINE_boolean('norm_class_attr', True, 'norm the class attributes')
# set up
tf.app.flags.DEFINE_integer('seed', '123', 'seed for reproduce')
tf.app.flags.DEFINE_integer('tf_seed', '123', 'seed for reproduce')
np.random.seed(FLAGS.seed)
tf.set_random_seed(FLAGS.tf_seed)
assert FLAGS.norm_class_attr == True, 'preprocessing only saved normed data'

if FLAGS.encoder == 'caffenet':
    enc_inputsize = 227
    enc_imagefeatsize = 4096
elif FLAGS.encoder == 'resnet':
    enc_inputsize = 224
    enc_imagefeatsize = 2048

# for the generator
if FLAGS.feat == 'fc6':
    recon_size = 256
    gen_imagefeatsize = 4096

upper_shape = {}
if FLAGS.dataset == 'cub':
    num_train_classes = cfg.CUB_TRAIN_CLASS_NUM
    num_classes = cfg.CUB_CLASS_NUM
    test1_total_iter = cfg.CUB_TEST_SEEN_NUM 
    test2_total_iter = cfg.CUB_TEST_UNSEEN_NUM
    upper_shape['cub'] = 'CUB'
    if FLAGS.semantic in ['glove', 'word2vec']:
        semantic_size = cfg.CUB_WORD2VEC_DIM # word2vec/glove
    elif FLAGS.semantic == 'class_attr':
        semantic_size = cfg.CUB_ATT_DIM
elif FLAGS.dataset == 'sun':
    num_train_classes = cfg.SUN_TRAIN_CLASS_NUM
    num_classes = cfg.SUN_CLASS_NUM
    test1_total_iter = cfg.SUN_TEST_SEEN_NUM
    test2_total_iter = cfg.SUN_TEST_UNSEEN_NUM
    upper_shape['sun'] = 'SUN'
    assert FLAGS.semantic == 'class_attr'
    semantic_size = cfg.SUN_ATT_DIM
elif FLAGS.dataset == 'apy':
    num_train_classes = cfg.APY_TRAIN_CLASS_NUM
    num_classes = cfg.APY_CLASS_NUM
    test1_total_iter = cfg.APY_TEST_SEEN_NUM
    test2_total_iter = cfg.APY_TEST_UNSEEN_NUM
    upper_shape['apy'] = 'APY'
    assert FLAGS.semantic == 'class_attr'
    semantic_size = cfg.APY_ATT_DIM
elif FLAGS.dataset == 'awa2':
    num_train_classes = cfg.AWA2_TRAIN_CLASS_NUM 
    num_classes = cfg.AWA2_CLASS_NUM
    test1_total_iter = cfg.AWA2_TEST_SEEN_NUM
    test2_total_iter = cfg.AWA2_TEST_UNSEEN_NUM
    upper_shape['awa2'] = 'AWA2'
    assert FLAGS.semantic == 'class_attr'
    semantic_size = cfg.AWA2_ATT_DIM 
else:
    raise NotImplementedError

data_path = os.path.join(cfg.PREPROCESSED_DATA_PATH, upper_shape[FLAGS.dataset], \
                            '%s_%s_h5'%(FLAGS.encoder, FLAGS.recon_encoder))
assert os.path.exists(data_path)

def main(_):
    try:
        assert len(_) == 1
    except AssertionError:
        print 'wrong parameters:'
        print _
        raise AssertionError
    dataset_info = save_data(extra_data=True)

    temp_summary_path = '%s_%s_%d_%.1f_%.3f_%d'%(FLAGS.dataset, FLAGS.mode, FLAGS.batch_size, FLAGS.keep_prob, FLAGS.margin, FLAGS.decay_curriculum)
    if FLAGS.summary_path:
        summary_path = os.path.join(FLAGS.summary, temp_summary_path + '_' + FLAGS.summary_path)
        checkpoint_path = os.path.join(FLAGS.checkpoint, temp_summary_path + '_' + FLAGS.summary_path)
        cachefile_path = os.path.join(FLAGS.cachefile, temp_summary_path + '_' + FLAGS.summary_path)
    else:
        summary_path = os.path.join(FLAGS.summary, temp_summary_path)
        checkpoint_path = os.path.join(FLAGS.checkpoint, temp_summary_path)
        cachefile_path = os.path.join(FLAGS.cachefile, temp_summary_path)

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth=True
    with tf.Session(config=gpu_config) as sess:

        sp_aen = SP_AEN(dataset_info=dataset_info,
                        enc_inputsize=enc_inputsize,
                        enc_imagefeatsize=enc_imagefeatsize,
                        gen_imagefeatsize=gen_imagefeatsize,
                        recon_size = recon_size,
                        num_train_classes=num_train_classes,
                        num_classes=num_classes, 
                        semantic_size=semantic_size,
                        summary_path=summary_path,
                        checkpoint_path=checkpoint_path,
                        cachefile_path=cachefile_path,
                        data_path = data_path)
        
        # initialize the network
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        sp_aen.load(sess)

        assert FLAGS.mode in ['trainf', 'traine', 'test']
        if FLAGS.mode == 'trainf':
            sp_aen.trainf(sess=sess)
        elif FLAGS.mode == 'traine':
            sp_aen.traine(sess=sess)
        elif FLAGS.mode == 'test':
            sp_aen.test(sess=sess)
        else:
            raise NotImplementedError


if __name__ == "__main__":
    tf.app.run()