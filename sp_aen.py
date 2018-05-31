import os
import scipy
from tqdm import tqdm
import numpy as np
import random
import h5py
import tensorflow as tf
from utils import patchShow
from data.preprocessing import read_data
from data.data_utils import norm_feat
from utils.util import hwc2chw, load_pretrained_model, fc
# from model.encoder import encoder_caffenet
# from model.resnet import encoder_resnet
from model.generator import generator_caffenet_fc6
from model.comparator import comparator_caffenet
# from model.classifier import classifier_caffenet
from model.discriminator_new import discriminator_cycle
from model.mapping_new import visual2semantic_2layer, semantic2visual_2layer, semantic2semantic_2layer
from utils.path import get_pretrain_generator, get_pretrain_comparator, get_pretrain_classifier

FLAGS = tf.app.flags.FLAGS

class ImagePool:
  """ History of generated images
      Same logic as https://github.com/junyanz/CycleGAN/blob/master/util/image_pool.lua
  """
  def __init__(self, pool_size):
    self.pool_size = pool_size
    self.images = []

  def query(self, image):
    if self.pool_size == 0:
      return image

    if len(self.images) < self.pool_size:
      self.images.append(image)
      return image
    else:
      p = random.random()
      if p > 0.5:
        # use old image
        random_id = random.randrange(0, self.pool_size)
        tmp = self.images[random_id].copy()
        self.images[random_id] = image.copy()
        return tmp
      else:
        return image

def _concat_mat(matrix_a, matrix_b):
    return tf.concat([matrix_a, matrix_b], axis=0)

def _split_mat(matrix):
    return tf.split(matrix, 2)

def _checkpath(path):
    assert os.path.exists(path)
    return path

class SP_AEN(object):
    def __init__(self, dataset_info, enc_inputsize, enc_imagefeatsize, gen_imagefeatsize, recon_size, num_train_classes, num_classes, \
                    semantic_size, summary_path, checkpoint_path, cachefile_path, data_path):
        margin = FLAGS.margin
        self.num_classes = num_classes
        self.dataset_info = dataset_info
        self.enc_inputsize = enc_inputsize
        self.enc_imagefeatsize = enc_imagefeatsize
        self.gen_imagefeatsize = gen_imagefeatsize
        self.semantic_size = semantic_size
        self.summary_path = summary_path
        self.checkpoint_path = checkpoint_path
        self.cachefile_path = cachefile_path

        self.train_feat_path = _checkpath(os.path.join(data_path, 'ps_trainval_feat.h5'))
        self.train_image_path = _checkpath(os.path.join(data_path, 'ps_trainval_image.h5'))
        self.test1_feat_path = _checkpath(os.path.join(data_path, 'ps_test_seen_feat.h5'))
        self.test1_image_path = _checkpath(os.path.join(data_path, 'ps_test_seen_image.h5'))
        self.test2_feat_path = _checkpath(os.path.join(data_path, 'ps_test_unseen_feat.h5'))
        self.test2_image_path = _checkpath(os.path.join(data_path, 'ps_test_unseen_image.h5'))

        self.global_step = tf.Variable(0, trainable=False)

        self.input_image = tf.placeholder(tf.float32, shape=(None, enc_inputsize, enc_inputsize, 3), name='input_image')
        self.image_feat = tf.placeholder(tf.float32, shape=(None, enc_imagefeatsize), name='image_feat')        
        self.recon_image_feat = tf.placeholder(tf.float32, shape=(None, gen_imagefeatsize), name='recon_image_feat')
        self.input_real_semantic = tf.placeholder(tf.float32, shape=(None, 1, semantic_size), name='input_real_semantic')
        self.input_fake_semantic = tf.placeholder(tf.float32, shape=(None, num_train_classes-1, semantic_size), name='input_fake_semantic')
        # self.pretrain_label = tf.placeholder(tf.int32, shape=(None,), name='pretrain_label')
        self.keep_prob = tf.placeholder(tf.float32, shape=())
        self.recon_topleft = ((recon_size - enc_inputsize)/2, (recon_size - enc_inputsize)/2)

        self.norm_image_feat = tf.nn.l2_normalize(self.image_feat, dim=1, name='norm_image_feat')
        self.rank_semantic, self.v2s_rank_vars = visual2semantic_2layer(self.norm_image_feat, enc_imagefeatsize, semantic_size, name='ranksemantic', \
                                                                        reuse=False, dropout=FLAGS.dropout, keep_prob=self.keep_prob)
        self.rank_semantic_sum = tf.summary.scalar('rank_semantic_norm', tf.reduce_mean(tf.norm(self.rank_semantic, axis=1)))
        self.map_rank_semantic, self.gan_F_vars = visual2semantic_2layer(self.recon_image_feat, gen_imagefeatsize, semantic_size, name='gan_F', reuse=False, dropout=False)
        self.map_recon_image_feat, self.gan_G_vars = semantic2visual_2layer(self.map_rank_semantic, semantic_size, gen_imagefeatsize, name='gan_G', reuse=False, dropout=False)

        self.map_rank_semantic_norm = tf.norm(self.map_rank_semantic, axis=1)
        self.L_map_rank = FLAGS.alpha_map_rank * tf.losses.mean_squared_error(labels=tf.ones_like(self.map_rank_semantic_norm)*FLAGS.map_rank_norm, predictions=self.map_rank_semantic_norm)        
        self.L_map_rank_sum = tf.summary.scalar('L_map_rank', self.L_map_rank / (FLAGS.alpha_map_rank + 1e-8))
        self.map_rank_semantic_sum = tf.summary.scalar('map_rank_semantic_norm', tf.reduce_mean(self.map_rank_semantic_norm))
        
        # Only for Visualization
        self.L_consist = FLAGS.alpha_consist * tf.losses.mean_squared_error(labels=self.recon_image_feat, predictions=self.map_recon_image_feat)
        self.L_consist_sum = tf.summary.scalar('L_consist', self.L_consist / (FLAGS.alpha_consist + 1e-8))

        self._real_rank_dis_logits, self.discriminator_rank_vars = discriminator_cycle(self.map_rank_semantic, name='rank', reuse=False)
        self._fake_rank_dis_logits, _ = discriminator_cycle(self.rank_semantic, name='rank', reuse=True)
        # Compare to original WGAN add sigmoid constrain
        self.real_rank_dis_logits = tf.sigmoid(self._real_rank_dis_logits)
        self.fake_rank_dis_logits = tf.sigmoid(self._fake_rank_dis_logits)

        wgan_alpha = tf.random_uniform(shape=[FLAGS.batch_size, 1], minval=0., maxval=1.0)
        wgan_differences = self.rank_semantic - self.map_rank_semantic
        wgan_interpolates = self.map_rank_semantic + (wgan_alpha * wgan_differences)
        interpolate_logits, _ = discriminator_cycle(wgan_interpolates, name='rank', reuse=True)
        wgan_gradients = tf.gradients(interpolate_logits, [wgan_interpolates])[0]
        wgan_slopes = tf.sqrt(tf.reduce_sum(tf.square(wgan_gradients), reduction_indices=[1]))
        wgan_gp = tf.reduce_mean((wgan_slopes-1.0)**2)

        self.L_rank_dis_wd =  FLAGS.alpha_rank_dis * (tf.reduce_mean(self.real_rank_dis_logits) - tf.reduce_mean(self.fake_rank_dis_logits)) 
        self.L_rank_dis_gp = FLAGS.alpha_rank_dis * FLAGS.wgan_lambda * wgan_gp
        self.L_rank_dis = -self.L_rank_dis_wd + self.L_rank_dis_gp
        self.L_rank_dis_wd_sum = tf.summary.scalar('L_rank_dis_wd', self.L_rank_dis_wd / (FLAGS.alpha_rank_dis + 1e-8))
        self.L_rank_dis_gp_sum = tf.summary.scalar('L_rank_dis_gp', self.L_rank_dis_gp / (FLAGS.alpha_rank_dis * FLAGS.wgan_lambda + 1e-8))

        self.L_rank_gen_E = -FLAGS.alpha_rank_gen * tf.reduce_mean(self.fake_rank_dis_logits)

        self.L_rank_dis_sum = tf.summary.scalar('L_rank_dis', self.L_rank_dis / (FLAGS.alpha_rank_dis + 1e-8))
        self.L_rank_gen_E_sum = tf.summary.scalar('L_rank_gen_E', self.L_rank_gen_E / (FLAGS.alpha_rank_gen + 1e-8))

        # Rank loss
        rank_positive_term = tf.reduce_sum(self.rank_semantic[:, None, :] * self.input_real_semantic, axis=2)
        rank_negative_term = tf.reduce_sum(self.rank_semantic[:, None, :] * self.input_fake_semantic, axis=2)
        all_rank_values = margin - rank_positive_term + rank_negative_term

        if FLAGS.rank_loss_type == 'mean':
            self.rank_loss = FLAGS.alpha_rank * tf.reduce_mean(tf.maximum(0.0, all_rank_values))
        elif FLAGS.rank_loss_type == 'random':
            break_rules = tf.cast((all_rank_values > 0), tf.float32)
            break_prob = (break_rules / tf.reduce_sum(break_rules, axis=1)[:, None])
            break_ind = tf.multinomial(tf.log(break_prob), 1)
            N = tf.cast(tf.shape(all_rank_values)[0], dtype=tf.int64) # for train, FLAGS.batch_size, for test, FLAGS.test_batch_size
            select_ind = tf.transpose(tf.concat([tf.range(N, dtype=tf.int64)[tf.newaxis, :], \
                                                 tf.squeeze(break_ind, [1])[tf.newaxis, :]], axis=0))
            select_rank_values = tf.gather_nd(params=all_rank_values, indices=select_ind)
            self.rank_loss = FLAGS.alpha_rank * tf.reduce_mean(tf.maximum(0.0, select_rank_values))
        elif FLAGS.rank_loss_type == 'max':
            max_rank_negative_term = tf.reduce_max(rank_negative_term, axis=1)
            self.rank_loss = FLAGS.alpha_rank * tf.reduce_mean(tf.maximum(0.0, margin - rank_positive_term + max_rank_negative_term))
        else:
            raise NotImplementedError

        # Generator
        if FLAGS.generator == 'caffenet' and FLAGS.feat == 'fc6':
            self._recon_image, self.generator_vars = generator_caffenet_fc6(self.map_recon_image_feat, reuse=False, trainable=False)
            self._recon_image = tf.reshape(self._recon_image, (-1, recon_size, recon_size, 3))
            self.recon_image = self._recon_image[:, self.recon_topleft[0]:self.recon_topleft[0]+enc_inputsize, 
                                                self.recon_topleft[1]:self.recon_topleft[1]+enc_inputsize, :]
        # Comparator
        if FLAGS.comparator == 'caffenet':
            if FLAGS.encoder == 'caffenet':         
                self.recon_image_comp, self.comparator_vars = comparator_caffenet(self.recon_image, reuse=False, trainable=False)
                self.input_image_comp, _ = comparator_caffenet(self.input_image, reuse=True, trainable=False)
            elif FLAGS.encoder == 'resnet':
                self.recon_image_comp, self.comparator_vars = comparator_caffenet(tf.image.resize_image_with_crop_or_pad(self.recon_image, 227, 227), reuse=False, trainable=False)
                self.input_image_comp, _ = comparator_caffenet(tf.image.resize_image_with_crop_or_pad(self.input_image, 227, 227), reuse=True, trainable=False)

        # # Classifier
        # if FLAGS.classifier == 'caffenet':
        #     if FLAGS.encoder == 'caffenet':
        #         self.recon_image_classfc8, self.classifier_vars= classifier_caffenet(self.recon_image, num_train_classes, reuse=False, trainable=False)
        #     elif FLAGS.encoder == 'resnet':
        #         self.recon_image_classfc8, self.classifier_vars = classifier_caffenet(tf.image.resize_image_with_crop_or_pad(self.recon_image, 227, 227), \
        #                                                                                 num_train_classes, reuse=False, trainable=False)

        print "Finish Build Model!"

        self.L_recon_img = FLAGS.alpha_recon_img * tf.losses.mean_squared_error(labels=self.input_image, predictions=self.recon_image) 
        self.L_recon_feat = FLAGS.alpha_recon_feat * tf.losses.mean_squared_error(labels=self.input_image_comp, predictions=self.recon_image_comp)
        # self.L_class = FLAGS.alpha_class * tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.pretrain_label, logits=(self.recon_image_classfc8 + 1e-8)))

        self.L_recon_img_sum = tf.summary.scalar('L_recon_img', self.L_recon_img / (FLAGS.alpha_recon_img + 1e-8))
        self.L_recon_feat_sum = tf.summary.scalar('L_recon_feat', self.L_recon_feat / (FLAGS.alpha_recon_feat + 1e-8))
        # self.L_class_sum = tf.summary.scalar('L_class', self.L_class / (FLAGS.alpha_class + 1e-8))
       
        self.rank_loss_sum = tf.summary.scalar('rank_loss', self.rank_loss / (FLAGS.alpha_rank + 1e-8))

        for var in self.v2s_rank_vars: 
            if 'weights' in var.op.name:
                tf.add_to_collection('rank_regularization', tf.nn.l2_loss(var))
        self.rank_regul_loss = FLAGS.alpha_rank_regularization * tf.add_n(tf.get_collection('rank_regularization'))

        for var in self.gan_G_vars+self.gan_F_vars:
            if 'weights' in var.op.name:
                tf.add_to_collection('gan_regularization', tf.nn.l2_loss(var))
        self.recon_regul_loss = FLAGS.alpha_gan_regularization * tf.add_n(tf.get_collection('gan_regularization'))

        self.map_loss = self.rank_loss + self.L_rank_gen_E + self.rank_regul_loss
        self.map_loss_sum = tf.summary.scalar('Map_Loss', self.map_loss)
        # self.cycle_loss = self.L_consist + self.L_map_rank + self.recon_regul_loss + self.L_rank_gen_F
        # self.cycle_loss = self.L_consist + self.L_map_rank + self.recon_regul_loss # self.L_rank_gen_F already set to 0
        self.cycle_loss = self.L_recon_img + self.L_recon_feat + self.L_map_rank + self.recon_regul_loss 
        self.cycle_loss_sum = tf.summary.scalar('Cycle_Loss', self.cycle_loss)
        self.total_loss = self.map_loss + self.cycle_loss
        self.total_loss_sum = tf.summary.scalar('Total_Loss', self.total_loss)

        self.L_E_SUM = tf.summary.merge([self.total_loss_sum, self.map_loss_sum, self.rank_loss_sum, self.L_rank_gen_E_sum])
        self.L_F_SUM = tf.summary.merge([self.total_loss_sum, self.cycle_loss_sum, self.L_consist_sum, self.L_map_rank_sum, self.L_recon_img_sum, self.L_recon_feat_sum])
        self.NORM_SUM = tf.summary.merge([self.rank_semantic_sum, self.map_rank_semantic_sum])
        self.DIS_SUM = tf.summary.merge([self.L_rank_dis_sum, self.L_rank_dis_wd_sum, self.L_rank_dis_gp_sum])

        # Optimizers
        lr_boundaries = [FLAGS.decay_curriculum]
        lr_values = [FLAGS.lr, FLAGS.lr*0.1]
        self.decayed_learning_rate = tf.train.piecewise_constant(self.global_step, lr_boundaries, lr_values)

        rank_trainer = tf.train.AdamOptimizer(self.decayed_learning_rate, beta1=FLAGS.beta1, beta2=FLAGS.beta2)
        rank_grads_and_vars = rank_trainer.compute_gradients(self.map_loss, self.v2s_rank_vars)
        rank_trainer_clip = [(tf.clip_by_norm(rank_op_grad, 10.), rank_op_var) for rank_op_grad, rank_op_var in rank_grads_and_vars]
        self.rank_optimizer = (rank_trainer.apply_gradients(rank_trainer_clip))

        recon_trainer = tf.train.AdamOptimizer(self.decayed_learning_rate, beta1=FLAGS.beta1, beta2=FLAGS.beta2)
        recon_grads_and_vars = recon_trainer.compute_gradients(self.cycle_loss, self.gan_G_vars+self.gan_F_vars)
        recon_trainer_clip = [(tf.clip_by_norm(recon_op_grad, 10.), recon_op_var) for recon_op_grad, recon_op_var in recon_grads_and_vars]
        self.recon_optimizer = (recon_trainer.apply_gradients(recon_trainer_clip))

        # joint_trainer = tf.train.AdamOptimizer(self.decayed_learning_rate, beta1=FLAGS.beta1, beta2=FLAGS.beta2)
        # joint_grads_and_vars = joint_trainer.compute_gradients(self.total_loss, self.v2s_rank_vars+self.gan_G_vars+self.gan_F_vars)
        # joint_trainer_clip = [(tf.clip_by_norm(joint_op_grad, 10.), joint_op_var) for joint_op_grad, joint_op_var in joint_grads_and_vars]
        # self.joint_optimizer = (joint_trainer.apply_gradients(joint_trainer_clip))

        dis_rank_trainer = tf.train.AdamOptimizer(self.decayed_learning_rate, beta1=FLAGS.beta1, beta2=FLAGS.beta2)
        dis_rank_grads_and_vars = dis_rank_trainer.compute_gradients(self.L_rank_dis, self.discriminator_rank_vars)
        dis_rank_trainer_clip = [(tf.clip_by_norm(dis_rank_op_grad, 10.), dis_rank_op_var) for dis_rank_op_grad, dis_rank_op_var in dis_rank_grads_and_vars]
        self.dis_rank_optimizer = (dis_rank_trainer.apply_gradients(dis_rank_trainer_clip))

        self.update_global_step = tf.assign(self.global_step, self.global_step+1)
        self.clear_global_step = tf.assign(self.global_step, 0)
        self.summary_writer = tf.summary.FileWriter(summary_path)

        generator_pretrain_vars = {}
        for var in self.generator_vars:
            assert 'generator' in var.op.name
            new_var_name = "/".join(var.op.name.split('/')[1:])
            generator_pretrain_vars[new_var_name] = var
        self.generator_saver = tf.train.Saver(generator_pretrain_vars)

        comparator_pretrain_vars = {}
        for var in self.comparator_vars:
            assert 'comparator' in var.op.name
            new_var_name = "/".join(var.op.name.split('/')[1:])
            comparator_pretrain_vars[new_var_name] = var
        self.comparator_saver = tf.train.Saver(comparator_pretrain_vars)

        # classifier_pretrain_vars = {}
        # for var in self.classifier_vars:
        #     assert 'classifier' in var.op.name
        #     new_var_name = "/".join(var.op.name.split('/')[1:])
        #     classifier_pretrain_vars[new_var_name] = var
        # self.classifier_saver = tf.train.Saver(classifier_pretrain_vars)

        # # debug
        # debug_vars = []
        # for var in tf.global_variables():
        #     if ('generator' in var.op.name) or ('comparator' in var.op.name):
        #         continue
        #     else:
        #         debug_vars.append(var)
        # self.debug_saver = tf.train.Saver(debug_vars)

        self.saver = tf.train.Saver(max_to_keep=100)

        self.unseen_test_mask = np.array(self.dataset_info['test_unseen_class_num'].keys())
        self.trainval_mask = np.array(self.dataset_info['trainval_class_num'].keys())

    def get_fake_semantic(self, all_fake_semantic, except_class_list):
        fake_semantic = []
        for class_i in except_class_list:
            temp_trainval_mask = self.trainval_mask
            fake_semantic.append(all_fake_semantic[np.delete(temp_trainval_mask,
                                                            np.where(temp_trainval_mask == class_i))-1])
        return np.array(fake_semantic)

    def get_RANK_minusbias(self, query_semantic, bias, train_mask, test_mask, semantic_type):
        if semantic_type == 'class_attr':
            assert not FLAGS.negative_class_attr
            if FLAGS.dataset == 'cub':
                dist_train = np.dot(query_semantic, self.dataset_info['positive_norm_class_attr'][train_mask-1].transpose())
                dist_test = np.dot(query_semantic, self.dataset_info['positive_norm_class_attr'][test_mask-1].transpose())
            else:
                dist_train = np.dot(query_semantic, self.dataset_info['class_attr'][train_mask-1].transpose())
                dist_test = np.dot(query_semantic, self.dataset_info['class_attr'][test_mask-1].transpose())
            dict_concate = np.concatenate([dist_train-bias, dist_test], axis=1)
            mask_concate = np.concatenate([train_mask, test_mask])
            return mask_concate[np.argmax(dict_concate, axis=1)]
        else:
            raise NotImplementedError

    def get_RANK(self, query_semantic, test_mask, semantic_type):
        if semantic_type == 'class_attr':
            if FLAGS.negative_class_attr:
                dist = np.dot(query_semantic, self.dataset_info['negative_norm_class_attr'][test_mask-1].transpose())
            else:
                if FLAGS.dataset == 'cub':
                    dist = np.dot(query_semantic, self.dataset_info['positive_norm_class_attr'][test_mask-1].transpose())                    
                else:
                    dist = np.dot(query_semantic, self.dataset_info['class_attr'][test_mask-1].transpose())
            return test_mask[np.argmax(dist, axis=1)]
        else:
            raise NotImplementedError

    # def compute_class_accuracy(self, total_number_dict, right_number_dict):
    #     class_accuracy = {}
    #     assert set(total_number_dict.keys()) == set(right_number_dict.keys())
    #     for key in total_number_dict.keys():
    #         assert key not in class_accuracy
    #         class_accuracy[key] = 1.0 * right_number_dict[key] / total_number_dict[key]
    #     return class_accuracy, np.mean(class_accuracy.values())

    def compute_class_accuracy_total(self, true_label, predict_label, classes):
        true_label = true_label[:, 0]
        nclass = len(classes)
        acc_per_class = np.zeros((nclass, 1))
        for i, class_i in enumerate(classes):
            idx = np.where(true_label == class_i)[0]
            acc_per_class[i] = (sum(true_label[idx] == predict_label[idx])*1.0 / len(idx))

        return np.mean(acc_per_class)

    def load(self, sess):
        print 'Load pretrained model >>>>>>>'
        # Load pretrained model
        generator_stats = tf.train.get_checkpoint_state(os.path.dirname(get_pretrain_generator(FLAGS.generator, load_type='tf')))
        if not generator_stats:
            load_pretrained_model('generator', get_pretrain_generator(FLAGS.generator, load_type='np'), sess, ignore_missing=True)
            self.generator_saver.save(sess, get_pretrain_generator(FLAGS.generator, load_type='tf'))
        self.generator_saver.restore(sess, get_pretrain_generator(FLAGS.generator, load_type='tf'))

        # load_pretrained_model('comparator', get_pretrain_comparator(FLAGS.comparator), sess, ignore_missing=True)
        comparator_stats = tf.train.get_checkpoint_state(os.path.dirname(get_pretrain_comparator(FLAGS.comparator, load_type='tf')))
        self.comparator_saver.restore(sess, comparator_stats.model_checkpoint_path)

        # #load_pretrained_model('classifier', get_pretrain_classifier(FLAGS.classifier), sess, ignore_missing=True)
        # classifier_stats = tf.train.get_checkpoint_state(os.path.dirname(get_pretrain_classifier(FLAGS.classifier, load_type='tf')))
        # if not classifier_stats:
        #     load_pretrained_model('classifier', get_pretrain_classifier(FLAGS.classifier, load_type='np'), sess, ignore_missing=True)
        #     self.classifier_saver.save(sess, get_pretrain_classifier(FLAGS.classifier, load_type='tf'))
        #     classifier_stats = tf.train.get_checkpoint_state(os.path.dirname(get_pretrain_classifier(FLAGS.classifier, load_type='tf')))
        # self.classifier_saver.restore(sess, classifier_stats.model_checkpoint_path)
        print "Finish load pretrained model to memory"

        print "Load data to memory >>>>>>>>>>>>"
        train_image_h5 = h5py.File(self.train_image_path, 'r')
        train_feat_h5 = h5py.File(self.train_feat_path, 'r')
        test1_image_h5 = h5py.File(self.test1_image_path, 'r')
        test1_feat_h5 = h5py.File(self.test1_feat_path, 'r')
        test2_image_h5 = h5py.File(self.test2_image_path, 'r')
        test2_feat_h5 = h5py.File(self.test2_feat_path, 'r')

        self.train_cl_list = train_feat_h5['class'][:]
        self.train_semantic_list = train_feat_h5['class_attr'][:]
        self.train_img_feat_list = train_feat_h5['image_feat'][:]
        self.train_recon_encoder_img_feat_list = train_feat_h5['recon_image_feat'][:]

        self.train_image_num = len(self.train_cl_list)
        self.train_img_feat_list = np.reshape(self.train_img_feat_list, (10*self.train_image_num, self.enc_imagefeatsize))
        self.train_recon_encoder_img_feat_list = np.reshape(self.train_recon_encoder_img_feat_list, (10*self.train_image_num, self.gen_imagefeatsize))
        self.train_sample_num = 10*self.train_image_num

        self.test1_cl_list = test1_feat_h5['class'][:]
        self.test1_semantic_list = test1_feat_h5['class_attr'][:]
        self.test1_img_feat_list = test1_feat_h5['image_feat'][:]
        self.test1_recon_encoder_img_feat_list = test1_feat_h5['recon_image_feat'][:]

        self.test2_cl_list = test2_feat_h5['class'][:]
        self.test2_semantic_list = test2_feat_h5['class_attr'][:]
        self.test2_img_feat_list = test2_feat_h5['image_feat'][:]
        self.test2_recon_encoder_img_feat_list = test2_feat_h5['recon_image_feat'][:]

        self.train_total_image = train_image_h5['image']
        self.test1_total_image = test1_image_h5['image']
        self.test2_total_image = test2_image_h5['image']
        print 'Finish load data to memory'

    def trainf(self, sess):

        # # debug
        # if FLAGS.retrain_model and (FLAGS.train_checkpoint != ''):
        #     try:
        #         ckpt_state = tf.train.get_checkpoint_state(FLAGS.train_checkpoint)
        #     except tf.errors.OutOfRangeError as e:
        #         tf.logging.error('Cannot restore checkpoint: %s', e)
        #     self.debug_saver.restore(sess, ckpt_state.model_checkpoint_path)
        #     print "Restore parameters from checkpoint!"
        # else:
        #     print "!!!No pretrained model load"

        if FLAGS.retrain_model and (FLAGS.train_checkpoint != ''):
            try:
                ckpt_state = tf.train.get_checkpoint_state(FLAGS.train_checkpoint)
            except tf.errors.OutOfRangeError as e:
                tf.logging.error('Cannot restore checkpoint: %s', e)
            self.saver.restore(sess, ckpt_state.model_checkpoint_path)
            print "Restore parameters from checkpoint!"

        sess.run([self.clear_global_step])

        if FLAGS.semantic == 'class_attr':
            if FLAGS.negative_class_attr:
                all_fake_semantic = self.dataset_info['negative_norm_class_attr']
            else:
                if FLAGS.dataset == 'cub':
                    all_fake_semantic = self.dataset_info['positive_norm_class_attr']
                else:
                    all_fake_semantic = self.dataset_info['class_attr']
        elif FLAGS.semantic == 'glove':
            all_fake_semantic = self.dataset_info['class_glove']
        elif FLAGS.semantic == 'word2vec':
            all_fake_semantic = self.dataset_info['class_word2vec']
        else:
            raise NotImplementedError

        read_offset = 0
        shuffle_sample_index = np.arange(self.train_sample_num)
        np.random.shuffle(shuffle_sample_index)
        for i in xrange(FLAGS.training_F_step):

            if i % FLAGS.ckpt_interval == 0  and i > 0:
                save_path = os.path.join(self.checkpoint_path, '%4d'%(i), FLAGS.model_name)
                print "save checkpoint in %s"%(save_path)
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                self.saver.save(sess, save_path, global_step=i)
  
            # read train batch data
            if read_offset + FLAGS.batch_size > self.train_sample_num:
                read_offset = 0
                np.random.shuffle(shuffle_sample_index)
            
            batch_sample_index = shuffle_sample_index[read_offset: read_offset+FLAGS.batch_size]
            batch_image_index = batch_sample_index / 10
            batch_crop_index = batch_sample_index % 10

            train_cl = self.train_cl_list[batch_image_index]
            train_img_feat = self.train_img_feat_list[batch_sample_index]
            train_semantic = self.train_semantic_list[batch_image_index]
            train_recon_img_feat = self.train_recon_encoder_img_feat_list[batch_sample_index]

            train_img_list = []
            read_offset = read_offset + FLAGS.batch_size

            for each_image_index, each_crop_index in zip(batch_image_index, batch_crop_index):
                train_img_list.append(self.train_total_image[each_image_index, each_crop_index])
            train_img = np.array(train_img_list)

            [_, _fake_rank_semantic, _train_loss, _L_F_SUM, _lr, _recon_regul_loss, _L_cycle_consist, _L_map_rank, _L_recon_img, _L_recon_feat] = \
                sess.run([self.recon_optimizer, self.map_rank_semantic, self.cycle_loss, self.L_F_SUM, \
                            self.decayed_learning_rate, self.recon_regul_loss, self.L_consist, self.L_map_rank, self.L_recon_img, self.L_recon_feat],
                            feed_dict={self.image_feat: train_img_feat,
                                       self.recon_image_feat: train_recon_img_feat,
                                       self.input_image: train_img,
                                       self.input_real_semantic: train_semantic[:, None, :],
                                       self.input_fake_semantic: self.get_fake_semantic(all_fake_semantic, train_cl),
                                       # self.pretrain_label: train_pretrain_cl1,
                                       self.keep_prob: FLAGS.keep_prob})

            self.summary_writer.add_summary(_L_F_SUM, i)

            if i % FLAGS.summary_interval == 0:
                [_NORM_SUM, _save_recon_image] = sess.run([self.NORM_SUM, self.recon_image],
                                                        feed_dict={self.image_feat: train_img_feat,
                                                                   self.recon_image_feat: train_recon_img_feat,
                                                                   self.input_image: train_img,
                                                                   self.input_real_semantic: train_semantic[:, None, :],
                                                                   self.input_fake_semantic: self.get_fake_semantic(all_fake_semantic, train_cl),
                                                                   # self.pretrain_label: train_pretrain_cl1,
                                                                   self.keep_prob: FLAGS.keep_prob})                
                self.summary_writer.add_summary(_NORM_SUM, i)


            if i % 1 == 0:
                print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
                print "Step %d, Total loss: %.3f, LR: %.6f, Consist loss: %.6f, Regul loss: %.6f, Map rank: %.6f, Image recon: %.6f, Feat recon: %.6f" \
                        %(i, _train_loss, _lr, _L_cycle_consist, _recon_regul_loss, _L_map_rank, _L_recon_img, _L_recon_feat)

     
            if i % FLAGS.saveimage_interval == 0:
                collage = patchShow.patchShow(np.concatenate((hwc2chw(train_img[:, :, :, ::-1]), hwc2chw(_save_recon_image[:, :, :, ::-1])), axis=3), in_range=(-120, 120))

                if not os.path.exists(self.cachefile_path):
                   os.makedirs(self.cachefile_path)
                scipy.misc.imsave(os.path.join(self.cachefile_path, 'reconstructions_%s_%s_%d.png'%(FLAGS.encoder, FLAGS.feat, i)), collage)
 
    def traine(self, sess):

        if FLAGS.retrain_model and (FLAGS.train_checkpoint != ''):
            try:
                ckpt_state = tf.train.get_checkpoint_state(FLAGS.train_checkpoint)
            except tf.errors.OutOfRangeError as e:
                tf.logging.error('Cannot restore checkpoint: %s', e)
            self.saver.restore(sess, ckpt_state.model_checkpoint_path)
            print "Restore parameters from checkpoint!"
        else:
            print "!!!No pretrained model load"

        sess.run([self.clear_global_step])

        if FLAGS.semantic == 'class_attr':
            if FLAGS.negative_class_attr:
                all_fake_semantic = self.dataset_info['negative_norm_class_attr']
            else:
                if FLAGS.dataset == 'cub':
                    all_fake_semantic = self.dataset_info['positive_norm_class_attr']
                else:
                    all_fake_semantic = self.dataset_info['class_attr']
        elif FLAGS.semantic == 'glove':
            all_fake_semantic = self.dataset_info['class_glove']
        elif FLAGS.semantic == 'word2vec':
            all_fake_semantic = self.dataset_info['class_word2vec']
        else:
            raise NotImplementedError

        read_offset = 0
        shuffle_sample_index = np.arange(self.train_sample_num)
        np.random.shuffle(shuffle_sample_index)
        for i in xrange(FLAGS.training_E_step):

            if i % FLAGS.ckpt_interval == 0  and i > 0:
                save_path = os.path.join(self.checkpoint_path, '%4d'%(i), FLAGS.model_name)
                print "save checkpoint in %s"%(save_path)
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                self.saver.save(sess, save_path, global_step=i)

            # read train batch data
            if read_offset + FLAGS.batch_size > self.train_sample_num:
                read_offset = 0
                np.random.shuffle(shuffle_sample_index)
            
            batch_sample_index = shuffle_sample_index[read_offset: read_offset+FLAGS.batch_size]
            batch_image_index = batch_sample_index / 10
            batch_crop_index = batch_sample_index % 10

            train_cl = self.train_cl_list[batch_image_index]
            train_img_feat = self.train_img_feat_list[batch_sample_index]
            train_semantic = self.train_semantic_list[batch_image_index]
            train_recon_img_feat = self.train_recon_encoder_img_feat_list[batch_sample_index]

            train_img_list = []
            read_offset = read_offset + FLAGS.batch_size

            for each_image_index, each_crop_index in zip(batch_image_index, batch_crop_index):
                train_img_list.append(self.train_total_image[each_image_index, each_crop_index])
            train_img = np.array(train_img_list)

            [_, _rank_semantic, _train_loss, _rank_loss, _L_E_SUM, _lr, _rank_regul_loss, _L_cycle_consist, _L_rank_gen_E] = \
                sess.run([self.rank_optimizer, self.rank_semantic, self.total_loss, self.rank_loss, self.L_E_SUM, self.decayed_learning_rate, \
                        self.rank_regul_loss, self.L_consist, self.L_rank_gen_E],
                                                        feed_dict={self.image_feat: train_img_feat,
                                                                   self.recon_image_feat: train_recon_img_feat,
                                                                   self.input_image: train_img,
                                                                   self.input_real_semantic: train_semantic[:, None, :],
                                                                   self.input_fake_semantic: self.get_fake_semantic(all_fake_semantic, train_cl),
                                                                   # self.pretrain_label: train_pretrain_cl1,
                                                                   self.keep_prob: FLAGS.keep_prob})

            for dummy_i in xrange(FLAGS.n_critics):
                # read train batch data
                if read_offset + FLAGS.batch_size > self.train_sample_num:
                    read_offset = 0
                    np.random.shuffle(shuffle_sample_index)
                
                batch_sample_index = shuffle_sample_index[read_offset: read_offset+FLAGS.batch_size]
                batch_image_index = batch_sample_index / 10
                batch_crop_index = batch_sample_index % 10

                train_cl = self.train_cl_list[batch_image_index]
                train_img_feat = self.train_img_feat_list[batch_sample_index]
                train_semantic = self.train_semantic_list[batch_image_index]
                train_recon_img_feat = self.train_recon_encoder_img_feat_list[batch_sample_index]

                train_img_list = []
                read_offset = read_offset + FLAGS.batch_size

                for each_image_index, each_crop_index in zip(batch_image_index, batch_crop_index):
                    train_img_list.append(self.train_total_image[each_image_index, each_crop_index])
                train_img = np.array(train_img_list)

                [_, _DIS_SUM, _L_rank_dis, _L_rank_dis_wd, _L_rank_dis_gp, _rank_semantic]= \
                    sess.run([self.dis_rank_optimizer, self.DIS_SUM, self.L_rank_dis, self.L_rank_dis_wd, self.L_rank_dis_gp, self.rank_semantic],
                                                            feed_dict={self.image_feat: train_img_feat,
                                                                       self.recon_image_feat: train_recon_img_feat,
                                                                       # self.recon_image_feat: train_recon_img_feat,
                                                                       self.input_image: train_img,
                                                                       self.input_real_semantic: train_semantic[:, None, :],
                                                                       self.input_fake_semantic: self.get_fake_semantic(all_fake_semantic, train_cl),
                                                                       # self.pretrain_label: train_pretrain_cl1, 
                                                                       self.keep_prob: FLAGS.keep_prob})

                self.summary_writer.add_summary(_DIS_SUM, i) # only add the last time of all n_critics times
 
            self.summary_writer.add_summary(_L_E_SUM, i)

            if i % FLAGS.summary_interval == 0:
                [_NORM_SUM] = sess.run([self.NORM_SUM],
                                        feed_dict={self.image_feat: train_img_feat,
                                                   self.recon_image_feat: train_recon_img_feat,
                                                   self.input_image: train_img,
                                                   self.input_real_semantic: train_semantic[:, None, :],
                                                   self.input_fake_semantic: self.get_fake_semantic(all_fake_semantic, train_cl),
                                                   # self.pretrain_label: train_pretrain_cl1,
                                                   self.keep_prob: FLAGS.keep_prob})               
                self.summary_writer.add_summary(_NORM_SUM, i)

            if i % 50 == 0:
                this_step_accuracy = np.mean(np.equal(self.get_RANK(_rank_semantic, np.arange(1, self.num_classes+1), FLAGS.semantic), train_cl))            
                print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
                print "Step %d, Total loss: %.3f, Rank loss: %.3f, LR: %.6f, Acc: %.6f "%(i, _train_loss, _rank_loss, _lr, this_step_accuracy)

                if FLAGS.debug:
                    print"Debug>>> Rank Regul loss: %.6f, Rank dis loss: %.6f"%(_rank_regul_loss, _L_rank_dis)
                    print"Debug>>> Rank dis wd: %.6f, Rand dis gp: %.6f, Gen E: %.6f"%(_L_rank_dis_wd, _L_rank_dis_gp, _L_rank_gen_E)

    def test(self, sess):
        assert FLAGS.test_checkpoint
        try:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.test_checkpoint)
        except tf.errors.OutOfRangeError as e:
            tf.logging.error('Cannot restore checkpoint: %s', e)
        self.saver.restore(sess, ckpt_state.model_checkpoint_path)
        print "Restore parameters from checkpoint!"

        sess.run([self.clear_global_step])

        if FLAGS.semantic == 'class_attr':
            if FLAGS.negative_class_attr:
                all_fake_semantic = self.dataset_info['negative_norm_class_attr']
            else:
                if FLAGS.dataset == 'cub':
                    all_fake_semantic = self.dataset_info['positive_norm_class_attr']
                else:
                    all_fake_semantic = self.dataset_info['class_attr']
        elif FLAGS.semantic == 'glove':
            all_fake_semantic = self.dataset_info['class_glove']
        elif FLAGS.semantic == 'word2vec':
            all_fake_semantic = self.dataset_info['class_word2vec']
        else:
            raise NotImplementedError

        zsl_test_unseen_total_acc = -1
        gzsl_test_unseen_total_acc = -1
        gzsl_test_seen_total_acc = -1
        [_test1_rank_semantic] = sess.run([self.rank_semantic],
                    feed_dict={self.image_feat: self.test1_img_feat_list,
                    self.recon_image_feat: self.test1_recon_encoder_img_feat_list,
                    self.input_real_semantic: self.test1_semantic_list[:, None, :],
                    self.keep_prob: 1.0})

        gzsl_test_seen_pred_label= self.get_RANK(_test1_rank_semantic, np.arange(1, self.num_classes+1), FLAGS.semantic)
        gzsl_test_seen_total_acc = self.compute_class_accuracy_total(self.test1_cl_list[:, None], gzsl_test_seen_pred_label, self.trainval_mask)

        print "Total Seen Test GZSL Accuracy: %.6f" %(gzsl_test_seen_total_acc)

        [_test2_rank_semantic] = sess.run([self.rank_semantic],
                    feed_dict={self.image_feat: self.test2_img_feat_list,
                    self.recon_image_feat: self.test2_recon_encoder_img_feat_list,
                    self.input_real_semantic: self.test2_semantic_list[:, None, :],
                    self.keep_prob: 1.0})

        zsl_test_unseen_pred_label = self.get_RANK(_test2_rank_semantic, self.unseen_test_mask, FLAGS.semantic)
        zsl_test_unseen_total_acc = self.compute_class_accuracy_total(self.test2_cl_list[:, None], zsl_test_unseen_pred_label, self.unseen_test_mask) 
        gzsl_test_unseen_pred_label = self.get_RANK(_test2_rank_semantic, np.arange(1, self.num_classes+1), FLAGS.semantic)
        gzsl_test_unseen_total_acc = self.compute_class_accuracy_total(self.test2_cl_list[:, None], gzsl_test_unseen_pred_label, self.unseen_test_mask)
        gzsl_H = (2*gzsl_test_unseen_total_acc*gzsl_test_seen_total_acc)/(gzsl_test_unseen_total_acc+gzsl_test_seen_total_acc+1e-8)
        print "Total Unseen Test ZSL Accuracy: %.6f, Total Unseen Test GZSL Accuracy: %.6f, GZSL H: %.6f" %(zsl_test_unseen_total_acc, gzsl_test_unseen_total_acc, gzsl_H)

  