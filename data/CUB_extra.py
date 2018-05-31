import sys
sys.path.append('../')
import cfg
import os
import scipy.io as io
from tqdm import tqdm
import numpy as np
import h5py
import tensorflow as tf
from data_utils import *
from model.resnet import encoder_resnet
from model.comparator import comparator_caffenet
from model.encoder import encoder_caffenet
from utils.path import get_pretrain_encoder, get_pretrain_comparator

FLAGS = tf.app.flags.FLAGS

def _get_class_attr_mean():
    # assert FLAGS.dataset == 'CUB'
    attribute_path = os.path.join('./data/CUB_200_2011/attributes.txt')
    with open(attribute_path, 'r') as f:
        data = f.readlines()

    all_attr_name=[]
    all_attr_ind=[]
    for attr_i in data:
        attr_i = attr_i.split('\n')[0].split('::')[0]
        ind, attr_name = attr_i.split(' ')
        if not attr_name in all_attr_name:
            all_attr_name.append(attr_name)
            all_attr_ind.append(int(ind)-1)    

    class_attr_mean = np.zeros((1, 312), dtype=np.float32)
    for start_i, end_i in zip(all_attr_ind, all_attr_ind[1:]+[312]):
        class_attr_mean[0, start_i:end_i] = 100.0 / (end_i - start_i)
         # print start_i, end_i

    return class_attr_mean

def _get_class_embedding(embedding_matrix, allclasses_file, class_attr_shape, class_word2vec_shape, class_glove_shape, class_wordnet_shape):
#    if FLAGS.dataset == 'cub':
    class_attr_cert = np.zeros(class_attr_shape, dtype=np.float32)
    class_word2vec = np.zeros(class_word2vec_shape, dtype=np.float32) 
    class_glove = np.zeros(class_glove_shape, dtype=np.float32)
    class_wordnet = np.zeros(class_wordnet_shape, dtype=np.float32)

    with open(allclasses_file) as f:
        allclasses = f.readlines()

    for ind, each_class in enumerate(allclasses):
        # each_class = each_class.split('\n')
        class_i = int(each_class.split('.')[0])

        if ind < 150:
            class_attr_cert[class_i-1] = embedding_matrix['trainval_cont'][:, ind]
            class_word2vec[class_i-1] = embedding_matrix['trainval_word2vec'][:, ind]
            class_glove[class_i-1] = embedding_matrix['trainval_glove'][:, ind]
            class_wordnet[class_i-1] = embedding_matrix['trainval_wordnet'][:, ind]
        else:
            class_attr_cert[class_i-1] = embedding_matrix['test_cont'][:, ind-150]
            class_word2vec[class_i-1] = embedding_matrix['test_word2vec'][:, ind-150]
            class_glove[class_i-1] = embedding_matrix['test_glove'][:, ind-150]
            class_wordnet[class_i-1] = embedding_matrix['test_wordnet'][:, ind-150]                

    return class_attr_cert, class_word2vec, class_glove, class_wordnet


def _read_class_attr(class_attr_path, class_attr_shape, negative=False, norm=True):
    with open(class_attr_path) as f:
        class_attributes = f.readlines()

    # Specific for CUB dataset
    class_attr = np.zeros(class_attr_shape, dtype=np.float32)
    for class_i, each_class_attr in enumerate(class_attributes):
        each_class_attr = each_class_attr.split('\n')[0]
        class_attr[class_i] = np.array(map(float, each_class_attr.split(' ')))
    if negative:
        class_attr_mean = _get_class_attr_mean()
        class_attr = class_attr - class_attr_mean
    if norm:
        class_attr = norm_feat(class_attr)

    return class_attr

def _read_image_attr(image_attr_path, image_attr_shape, norm=False):
    with open(image_attr_path) as f:
        all_image_attr = f.readlines()

    image_attr = np.zeros(image_attr_shape, dtype=np.float32)
    for each_image_attr in all_image_attr:
        each_image_attr = each_image_attr.split('\n')[0]
        each_image_attr = map(int, each_image_attr.split(' ')[:3]) # time can not change into int type
        image_attr[each_image_attr[0]-1, each_image_attr[1]-1] = each_image_attr[2]
    if norm:
        image_attr = norm_feat(image_attr)
            
    return image_attr

def _read_class(datapath):
    with open(os.path.join(datapath, 'classes.txt')) as f:
        all_classes = f.readlines()

    class2index = {}
    index2class = {}
    for each_class in all_classes:
        index_i, class_i = each_class.split(' ')
        class_i = class_i.split('\n')[0]
        class2index[class_i] = int(index_i)
        index2class[int(index_i)] = class_i

    return class2index, index2class

def _read_images(datapath):
    """
    class2images: return dict{class1:[], class2:[]}
    """
    with open(os.path.join(datapath, 'images.txt')) as f:
        all_imagepath = f.readlines() 

    class2images = {}
    all_images = {}
    index2image = {}
    image2index = {}
    for i, each_image in tqdm(enumerate(all_imagepath, 1)):
        each_image = each_image.split('\n')[0]
        num_i, each_image = each_image.split(' ')
            
        assert i == float(num_i)
        index2image[i] = each_image
        image2index[os.path.basename(each_image)] = i
        class_name = os.path.dirname(each_image)
        all_images[each_image] = {'index': i, 'class': class_name}
            
        if class_name not in class2images:
            class2images[class_name] = []
        class2images[class_name].append(each_image)
    print "Read all images already!"
    return class2images, all_images, index2image, image2index

def convert_to_tfrecords(writer, imageinfo, batch_enc_image_feat, batch_recon_image_feat, batch_image, encoder_net_inputsize):
    for each_enc_image_feat, each_recon_image_feat, each_image in zip(batch_enc_image_feat, batch_recon_image_feat, batch_image):
        each_image_raw = each_image.reshape((-1, encoder_net_inputsize*encoder_net_inputsize*3)).tobytes()
        each_recon_image_feat = each_recon_image_feat.reshape((4096,))
        example = tf.train.Example(features=tf.train.Features(feature={
                'index': int64_feature([imageinfo['index']]),
                'class': int64_feature([imageinfo['class']]),
                'pretrain_class': int64_feature([imageinfo['pretrain_class']]),
                'positive_norm_class_attr': float_feature(imageinfo['positive_norm_class_attr']),
                'negative_norm_class_attr': float_feature(imageinfo['negative_norm_class_attr']),
                'class_word2vec': float_feature(imageinfo['class_word2vec']),
                'class_glove': float_feature(imageinfo['class_glove']),
                'class_wordnet': float_feature(imageinfo['class_wordnet']),
                'image_feat':float_feature(each_enc_image_feat),
                'recon_image_feat':float_feature(each_recon_image_feat),
                'image':bytes_feature(each_image_raw)
            }))
        # print example
        writer.write(example.SerializeToString())

def save_cub_extra_data(save_tfrecords=False, save_type='tf'):
    attr_dim = cfg.CUB_ATT_DIM
    class_num = cfg.CUB_CLASS_NUM
    image_num = cfg.CUB_IMAGE_NUM
    word2vec_dim = cfg.CUB_WORD2VEC_DIM
    glove_dim = cfg.CUB_GLOVE_DIM
    wordnet_dim = cfg.CUB_WORDNET_DIM
    # ss_split_path = os.path.join(cfg.SS_SPLIT_PATH, 'CUB')
    ps_split_path = os.path.join(cfg.PS_SPLIT_PATH, 'CUB')
    save_data_path = checkdir(os.path.join(cfg.PREPROCESSED_DATA_PATH, 'CUB'))
    image_path = os.path.join(cfg.CUB_PATH, 'images')

    data_path = cfg.CUB_PATH
    class_attr_path = os.path.join(cfg.CUB_PATH, './attributes/class_attribute_labels_continuous.txt')
    image_attr_path = os.path.join(cfg.CUB_PATH, './attributes/image_attribute_labels.txt')

    # ss_attr_data = io.loadmat(checkfile(os.path.join(ss_split_path, 'att_splits.mat')))
    ps_attr_data = io.loadmat(checkfile(os.path.join(ps_split_path, 'att_splits.mat')))
    ps_image_filenames = io.loadmat(checkfile(os.path.join(ps_split_path, 'ps_image_files.mat')))['image_files']
    ps_image_filenames_list = convert_mat_to_list(ps_image_filenames, 'cub')

    embedding_data = io.loadmat(checkfile(os.path.join(os.path.dirname(cfg.CUB_PATH), 'CUB_embedding.mat')))

    class2index, index2class = _read_class(data_path)
    class2images, allimages, index2image, image2index = _read_images(data_path)

    negative_norm_class_attr = _read_class_attr(class_attr_path, class_attr_shape=[class_num, attr_dim], negative=True, norm=True)
    positive_norm_class_attr = _read_class_attr(class_attr_path, class_attr_shape=[class_num, attr_dim], negative=False, norm=True)

    image_attr = _read_image_attr(image_attr_path, image_attr_shape=[image_num, attr_dim], norm=False)


    # Get class label embedding(word2vec, glove, wordnet)
    class_attr_cert, class_word2vec, class_glove, class_wordnet = _get_class_embedding(embedding_data, 
                                                                checkfile(os.path.join(ps_split_path, 'allclasses.txt')),
                                                                class_attr_shape=[class_num, attr_dim],
                                                                class_word2vec_shape=[class_num, word2vec_dim],
                                                                class_glove_shape=[class_num, glove_dim],
                                                                class_wordnet_shape=[class_num, wordnet_dim])
    assert np.all(np.allclose(positive_norm_class_attr, class_attr_cert))


    ps_trainval_index, ps_test_unseen_index, ps_test_seen_index, \
        ps_train_index, ps_val_index  = ps_attr_data['trainval_loc'], \
        ps_attr_data['test_unseen_loc'], ps_attr_data['test_seen_loc'], \
        ps_attr_data['train_loc'], ps_attr_data['val_loc']

    def _get_imageinfo(mat_index):
        assert mat_index.ndim == 2
        mat_index = mat_index[:, 0].tolist()
        ret_images = []
        for i in mat_index:
            ps_image_path =  ps_image_filenames_list[i-1] # origin matlab index start from 1

            each_image = {}
            each_image['index'] = image2index[ps_image_path]
            each_image['path'] = index2image[each_image['index']]
            assert os.path.basename(each_image['path']) == ps_image_path
            each_image['class'] = class2index[allimages[each_image['path']]['class']] # start from 1

            each_image['positive_norm_class_attr'] = positive_norm_class_attr[each_image['class']-1]
            each_image['negative_norm_class_attr'] = negative_norm_class_attr[each_image['class']-1]
            each_image['class_word2vec'] = class_word2vec[each_image['class']-1]
            each_image['class_glove'] = class_glove[each_image['class']-1]
            each_image['class_wordnet'] = class_wordnet[each_image['class']-1]
            each_image['image_attr'] = image_attr[i-1] # minus 1 for index start from 1 in i
            ret_images.append(each_image)
        return ret_images

    # Get image path for each proposed split
    [ps_trainval_images, ps_test_unseen_images, ps_test_seen_images,
        ps_train_images, ps_val_images] = map(_get_imageinfo, [ps_trainval_index,
            ps_test_unseen_index, ps_test_seen_index, ps_train_index, ps_val_index])

    def _create_pretrain_dict(image_list):
        class_list = []
        for image in image_list:
            if not index2class[image['class']] in class_list:
                class_list.append(index2class[image['class']])
        pretrainclass2index = {}
        index2pretrainclass = {}
        for k, v in enumerate(class_list):
            pretrainclass2index[v] = k
            index2pretrainclass[k] = v
        return pretrainclass2index, index2pretrainclass

    pretrainclass2index, index2pretrainclass = _create_pretrain_dict(ps_trainval_images)

    def _get_split_class_number(image_list):
        class_list = {}
        for image in image_list:
            if not image['class'] in class_list:
                class_list[image['class']] = 1
            else:
                class_list[image['class']] += 1
        return class_list

    [trainval_class_number, train_class_number, val_class_number, test_seen_class_number, \
         test_unseen_class_number] = map(_get_split_class_number, [ps_trainval_images, ps_train_images, \
            ps_val_images, ps_test_seen_images, ps_test_unseen_images])
    
    def _add_pretrain_class(image_list):
        for image in image_list:
            image['pretrain_class'] = pretrainclass2index[index2class[image['class']]]
        return image_list

    def _add_dummy_pretrain_class(image_list):
        for image in image_list:
            image['pretrain_class'] = -1
        return image_list

    [ps_trainval_images, ps_test_seen_images, ps_train_images, ps_val_images] = map(_add_pretrain_class, 
                        [ps_trainval_images, ps_test_seen_images, ps_train_images, ps_val_images])

    [ps_test_unseen_images] = map(_add_dummy_pretrain_class, [ps_test_unseen_images])

    if save_tfrecords:
        print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        print "Start to store image information to TFRecord"
        print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

        gpu_config = tf.ConfigProto()
        # gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.8
        gpu_config.gpu_options.allow_growth=True
        with tf.Session(config=gpu_config) as sess:
            assert FLAGS.encoder == 'resnet'
            encoder_net_inputsize = 224
            input_image = tf.placeholder(tf.float32, shape=(None, encoder_net_inputsize, encoder_net_inputsize, 3), name='input_image')
            output_feat, resnet_vars = encoder_resnet(input_image, layer=101, reuse=False)

            resnet_pretrain_vars = {}
            for var in resnet_vars:
                assert 'encoder' in var.op.name
                new_var_name = "/".join(var.op.name.split('/')[1:])
                resnet_pretrain_vars[new_var_name] = var
            resnet_saver  = tf.train.Saver(resnet_pretrain_vars)           

            resnet_saver.restore(sess, get_pretrain_encoder('resnet', load_type='tf'))
            print 'Finsh load pretrained encoder resnet model'


            # get imagenet mean, but in BGR not RGB
            matfile = io.loadmat(os.path.join(cfg.PRETRAIN_MODEL, 'ilsvrc_2012_mean.mat'))
            image_mean = matfile['image_mean']
            topleft = ((image_mean.shape[0] - encoder_net_inputsize)/2, (image_mean.shape[1] - encoder_net_inputsize)/2)
            crop_image_mean = image_mean[topleft[0]:topleft[0]+encoder_net_inputsize, topleft[1]:topleft[1]+encoder_net_inputsize]

            assert FLAGS.recon_encoder == 'caffenet'
            recon_net_inputsize = 227
            recon_encoder_input_image = tf.placeholder(tf.float32, shape=(None, recon_net_inputsize, recon_net_inputsize, 3), name='recon_input_image')
            recon_encoder_output_feat, recon_encoder_vars = encoder_caffenet(recon_encoder_input_image, feat='fc6', reuse=False, trainable=False)

            recon_encoder_pretrain_vars = {}
            for var in recon_encoder_vars:
                assert 'encoder_caffenet' in var.op.name
                new_var_name = "/".join(var.op.name.split('/')[1:])
                recon_encoder_pretrain_vars[new_var_name] = var
            recon_encoder_saver = tf.train.Saver(recon_encoder_pretrain_vars)
     
            recon_encoder_stats = tf.train.get_checkpoint_state(os.path.dirname(get_pretrain_encoder(FLAGS.recon_encoder, load_type='tf')))
            recon_encoder_saver.restore(sess, recon_encoder_stats.model_checkpoint_path)       
            print 'Finish load pretrained recon encoder caffenet model'
        
            save_splits_list_train = ['ps_trainval', 'ps_train']
            save_splits_list_test = ['ps_test_unseen', 'ps_test_seen', 'ps_val']

            if save_type == 'tf':
                for each_save_split in save_splits_list_train:
                    # tfrecords_save_path = os.path.join(save_data_path, 'debug', each_save_split+'.tfrecords')
                    checkdir(os.path.join(save_data_path, '%s_%s_extra'%(FLAGS.encoder, FLAGS.recon_encoder)))
                    tfrecords_save_path = os.path.join(save_data_path, '%s_%s_extra'%(FLAGS.encoder, FLAGS.recon_encoder), each_save_split+'.tfrecords')
                    if os.path.exists(tfrecords_save_path):
                        print "Skip store split %s"%(tfrecords_save_path)
                        continue
                    writer = tf.python_io.TFRecordWriter(tfrecords_save_path)
                    for each_image in tqdm(eval(each_save_split + '_images')):
                        batch_crop_each_image = get_ten_crop(each_image, image_path, encoder_net_inputsize)
                        # preprocessing images
                        batch_crop_each_image = batch_crop_each_image[:, :, :, ::-1]
                        batch_crop_each_image_minus_mean = batch_crop_each_image - crop_image_mean                

                        [batch_crop_enc_each_feat] = sess.run([output_feat], feed_dict={input_image: batch_crop_each_image_minus_mean})

                        batch_recon_encoder_input_image = np.zeros((10, recon_net_inputsize, recon_net_inputsize, 3))
                        batch_recon_encoder_input_image[:, 1:1+encoder_net_inputsize, 1:1+encoder_net_inputsize, :] = batch_crop_each_image_minus_mean
                        [batch_crop_recon_each_feat] = sess.run([recon_encoder_output_feat], feed_dict={recon_encoder_input_image: batch_recon_encoder_input_image})
                        convert_to_tfrecords(writer, each_image, batch_crop_enc_each_feat, batch_crop_recon_each_feat, 
                                                batch_crop_each_image, encoder_net_inputsize)
                    writer.close()
                    print('Finish store split %s'%(each_save_split))

                for each_save_split in save_splits_list_test:
                    # tfrecords_save_path = os.path.join(save_data_path, 'debug', each_save_split+'.tfrecords')
                    checkdir(os.path.join(save_data_path, '%s_%s_extra'%(FLAGS.encoder, FLAGS.recon_encoder)))
                    tfrecords_save_path = os.path.join(save_data_path, '%s_%s_extra'%(FLAGS.encoder, FLAGS.recon_encoder), each_save_split+'.tfrecords')
                    if os.path.exists(tfrecords_save_path):
                        print "Skip store split %s"%(tfrecords_save_path)
                        continue
                    writer = tf.python_io.TFRecordWriter(tfrecords_save_path)
                    for each_image in tqdm(eval(each_save_split + '_images')):
                        batch_crop_each_image = get_test_crop(each_image, image_path, encoder_net_inputsize)
                        # preprocessing images
                        batch_crop_each_image = batch_crop_each_image[:, :, ::-1][None, :, :, :]
                        batch_crop_each_image_minus_mean = batch_crop_each_image - crop_image_mean                

                        [batch_crop_enc_each_feat] = sess.run([output_feat], feed_dict={input_image: batch_crop_each_image_minus_mean})
                        
                        batch_recon_encoder_input_image = np.zeros((1, recon_net_inputsize, recon_net_inputsize, 3))
                        batch_recon_encoder_input_image[:, 1:1+encoder_net_inputsize, 1:1+encoder_net_inputsize, :] = batch_crop_each_image_minus_mean
                        [batch_crop_recon_each_feat] = sess.run([recon_encoder_output_feat], feed_dict={recon_encoder_input_image: batch_recon_encoder_input_image})
                        convert_to_tfrecords(writer, each_image, batch_crop_enc_each_feat, batch_crop_recon_each_feat,
                                                batch_crop_each_image, encoder_net_inputsize)
                    writer.close()
                    print('Finish store split %s'%(each_save_split))

            elif save_type  == 'h5':
                for each_save_split in save_splits_list_train:
                    checkdir(os.path.join(save_data_path, '%s_%s_h5'%(FLAGS.encoder, FLAGS.recon_encoder)))
                    h5_image_save_path = os.path.join(save_data_path, '%s_%s_h5'%(FLAGS.encoder, FLAGS.recon_encoder), each_save_split+'_image.h5')
                    h5_feat_save_path = os.path.join(save_data_path, '%s_%s_h5'%(FLAGS.encoder, FLAGS.recon_encoder), each_save_split+'_feat.h5')

                    if os.path.exists(h5_image_save_path) and os.path.exists(h5_feat_save_path):
                        print "Skip store split %s and %s"%(h5_image_save_path, h5_feat_save_path)
                        continue   
                    else:
                        split_N = len(eval(each_save_split + '_images'))

                        h5_image_file = h5py.File(h5_image_save_path, 'w')
                        h5_feat_file = h5py.File(h5_feat_save_path, 'w')
                        h5_feat_file.create_dataset('index', (split_N, ), dtype=np.int32)
                        h5_feat_file.create_dataset('class', (split_N, ), dtype=np.int32)
                        h5_feat_file.create_dataset('class_attr', (split_N, attr_dim), dtype=np.float32)
                        h5_feat_file.create_dataset('class_word2vec', (split_N, word2vec_dim), dtype=np.float32)
                        h5_feat_file.create_dataset('class_glove', (split_N, glove_dim), dtype=np.float32)
                        h5_feat_file.create_dataset('image_feat', (split_N, 10, 2048), dtype=np.float32)
                        h5_feat_file.create_dataset('recon_image_feat', (split_N, 10, 4096), dtype=np.float32)

                        h5_image_file.create_dataset('image', (split_N, 10, 224, 224, 3), dtype=np.float32)

                        split_image_index_dict = {}
                        for each_image_i, each_image in tqdm(enumerate(eval(each_save_split + '_images'))):

                            batch_crop_each_image = get_ten_crop(each_image, image_path, encoder_net_inputsize)
                            # preprocessing images
                            batch_crop_each_image = batch_crop_each_image[:, :, :, ::-1]
                            batch_crop_each_image_minus_mean = batch_crop_each_image - crop_image_mean                

                            [batch_crop_enc_each_feat] = sess.run([output_feat], feed_dict={input_image: batch_crop_each_image_minus_mean})

                            batch_recon_encoder_input_image = np.zeros((10, recon_net_inputsize, recon_net_inputsize, 3))
                            batch_recon_encoder_input_image[:, 1:1+encoder_net_inputsize, 1:1+encoder_net_inputsize, :] = batch_crop_each_image_minus_mean
                            [batch_crop_recon_each_feat] = sess.run([recon_encoder_output_feat], feed_dict={recon_encoder_input_image: batch_recon_encoder_input_image})

                            h5_feat_file['index'][each_image_i] = each_image['index']
                            h5_feat_file['class'][each_image_i] = each_image['class']
                            h5_feat_file['class_attr'][each_image_i] = each_image['positive_norm_class_attr']
                            h5_feat_file['class_word2vec'][each_image_i] = each_image['class_word2vec']
                            h5_feat_file['class_glove'][each_image_i] = each_image['class_glove']
                            h5_feat_file['image_feat'][each_image_i] = batch_crop_enc_each_feat
                            h5_feat_file['recon_image_feat'][each_image_i] = batch_crop_recon_each_feat

                            # h5_image_file['image'][each_image_i] = batch_crop_each_image
                            h5_image_file['image'][each_image_i] = batch_crop_each_image_minus_mean

                        h5_image_file.close()
                        h5_feat_file.close()
                        print "Finish save %s and %s."%(h5_image_save_path, h5_feat_save_path)

                for each_save_split in save_splits_list_test:
                    checkdir(os.path.join(save_data_path, '%s_%s_h5'%(FLAGS.encoder, FLAGS.recon_encoder)))
                    h5_image_save_path = os.path.join(save_data_path, '%s_%s_h5'%(FLAGS.encoder, FLAGS.recon_encoder), each_save_split+'_image.h5')
                    h5_feat_save_path = os.path.join(save_data_path, '%s_%s_h5'%(FLAGS.encoder, FLAGS.recon_encoder), each_save_split+'_feat.h5')

                    if os.path.exists(h5_image_save_path) and os.path.exists(h5_feat_save_path):
                        print "Skip store split %s and %s"%(h5_image_save_path, h5_feat_save_path)
                        continue   
                    else:
                        split_N = len(eval(each_save_split + '_images'))

                        h5_image_file = h5py.File(h5_image_save_path, 'w')
                        h5_feat_file = h5py.File(h5_feat_save_path, 'w')
                        h5_feat_file.create_dataset('index', (split_N, ), dtype=np.int32)
                        h5_feat_file.create_dataset('class', (split_N, ), dtype=np.int32)
                        h5_feat_file.create_dataset('class_attr', (split_N, attr_dim), dtype=np.float32)
                        h5_feat_file.create_dataset('class_word2vec', (split_N, word2vec_dim), dtype=np.float32)
                        h5_feat_file.create_dataset('class_glove', (split_N, word2vec_dim), dtype=np.float32)
                        h5_feat_file.create_dataset('image_feat', (split_N, 2048), dtype=np.float32)
                        h5_feat_file.create_dataset('recon_image_feat', (split_N, 4096), dtype=np.float32)

                        h5_image_file.create_dataset('image', (split_N, 224, 224, 3), dtype=np.float32)

                        split_image_index_dict = {}
                        for each_image_i, each_image in tqdm(enumerate(eval(each_save_split + '_images'))):

                            batch_crop_each_image = get_test_crop(each_image, image_path, encoder_net_inputsize)
                            # preprocessing images
                            batch_crop_each_image = batch_crop_each_image[:, :, ::-1][None, :, :, :]
                            batch_crop_each_image_minus_mean = batch_crop_each_image - crop_image_mean                

                            [batch_crop_enc_each_feat] = sess.run([output_feat], feed_dict={input_image: batch_crop_each_image_minus_mean})

                            batch_recon_encoder_input_image = np.zeros((1, recon_net_inputsize, recon_net_inputsize, 3))
                            batch_recon_encoder_input_image[:, 1:1+encoder_net_inputsize, 1:1+encoder_net_inputsize, :] = batch_crop_each_image_minus_mean
                            [batch_crop_recon_each_feat] = sess.run([recon_encoder_output_feat], feed_dict={recon_encoder_input_image: batch_recon_encoder_input_image})

                            h5_feat_file['index'][each_image_i] = each_image['index']
                            h5_feat_file['class'][each_image_i] = each_image['class']
                            h5_feat_file['class_attr'][each_image_i] = each_image['positive_norm_class_attr']
                            h5_feat_file['class_word2vec'][each_image_i] = each_image['class_word2vec']
                            h5_feat_file['class_glove'][each_image_i] = each_image['class_glove']
                            h5_feat_file['image_feat'][each_image_i] = batch_crop_enc_each_feat
                            h5_feat_file['recon_image_feat'][each_image_i] = batch_crop_recon_each_feat

                            # h5_image_file['image'][each_image_i] = batch_crop_each_image
                            h5_image_file['image'][each_image_i] = batch_crop_each_image_minus_mean

                        h5_image_file.close()
                        h5_feat_file.close()
                        print "Finish save %s and %s."%(h5_image_save_path, h5_feat_save_path)

                

    return {'class2index':class2index, 'index2class':index2class, 
            'class2images':class2images, 'allimages':allimages, 'index2image':index2image, 
            'pretrainclass2index': pretrainclass2index, 
            'index2pretrainclass': index2pretrainclass,
            'positive_norm_class_attr':positive_norm_class_attr, 
            'negative_norm_class_attr':negative_norm_class_attr,
            'image_attr': image_attr, 'class_word2vec':class_word2vec,
            'class_glove': class_glove, 'class_wordnet': class_wordnet,
            'trainval_class_num': trainval_class_number,
            'train_class_num': train_class_number,
            'val_class_num': val_class_number,
            'test_seen_class_num': test_seen_class_number,
            'test_unseen_class_num': test_unseen_class_number,
            'ps_trainval_images': ps_trainval_images,
            'ps_test_unseen_images': ps_test_unseen_images,
            'image_attr': image_attr}


def read_cub_extra_data(filename_queue, batch_size, image_mean, mode):
    attr_dim = cfg.CUB_ATT_DIM
    word2vec_dim = cfg.CUB_WORD2VEC_DIM
    glove_dim = cfg.CUB_GLOVE_DIM
    wordnet_dim = cfg.CUB_WORDNET_DIM

    assert FLAGS.encoder == 'resnet'
    feat_dim = cfg.RESNET_FEAT_DIM
    encoder_net_inputsize = 224
    
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
            serialized_example,
            features = {
                'index': tf.FixedLenFeature([], tf.int64),
                'class': tf.FixedLenFeature([], tf.int64),
                'pretrain_class': tf.FixedLenFeature([], tf.int64),
                'positive_norm_class_attr': tf.FixedLenFeature([attr_dim], tf.float32),
                'negative_norm_class_attr': tf.FixedLenFeature([attr_dim], tf.float32),
                'class_word2vec': tf.FixedLenFeature([word2vec_dim], tf.float32),
                'class_glove': tf.FixedLenFeature([glove_dim], tf.float32),
                'class_wordnet': tf.FixedLenFeature([wordnet_dim], tf.float32),
                'image_feat': tf.FixedLenFeature([feat_dim], tf.float32),
                'recon_image_feat': tf.FixedLenFeature([4096], tf.float32),
                'image': tf.FixedLenFeature([encoder_net_inputsize*encoder_net_inputsize*3], tf.string)
            }
        )

    each_index = tf.cast(features['index'], tf.int32)
    each_class = tf.cast(features['class'], tf.int32)
    each_pretrain_class = tf.cast(features['pretrain_class'], tf.int32)
    each_positive_norm_class_attr = features['positive_norm_class_attr']
    each_negative_norm_class_attr = features['negative_norm_class_attr']
    each_class_word2vec = features['class_word2vec']
    each_class_glove = features['class_glove']
    each_class_wordnet = features['class_wordnet']

    each_image_feat = features['image_feat']
    each_recon_image_feat = features['recon_image_feat']

    temp_image = tf.decode_raw(features['image'], tf.uint8)
    each_image = tf.reshape(temp_image, [encoder_net_inputsize, encoder_net_inputsize, 3])
    each_image = tf.cast(each_image, tf.float32)

    each_image = each_image - image_mean
    # # preprocess the image
    # _image = tf.decode_raw(features['image'], tf.uint8)
    # each_image = tf.reshape(_image, [cfg.SAVE_IMAGE_HEIGHT, cfg.SAVE_IMAGE_WIDTH, 3])
    # each_image = tf.cast(each_image, tf.float32)

    # if mode == 'train':
    #     each_image = tf.random_crop(each_image, [net_inputsize, net_inputsize, 3])
    #     each_image = tf.image.random_flip_left_right(each_image)
    # elif mode == 'test':
    #     each_image = tf.image.resize_images(each_image, [net_inputsize, net_inputsize])

    # # convert RGB to BGR
    # each_image = each_image[:, :, ::-1]
    # each_image = each_image - image_mean

    if mode == 'train':
        _index, _class, _pretrain_class, _positive_norm_class_attr, _negative_norm_class_attr, \
            _class_word2vec, _class_glove, _class_wordnet, _image_feat, _recon_image_feat, _image \
                = tf.train.shuffle_batch([each_index, each_class, each_pretrain_class, 
                                          each_positive_norm_class_attr, 
                                          each_negative_norm_class_attr,
                                          each_class_word2vec, each_class_glove, each_class_wordnet,
                                          each_image_feat, each_recon_image_feat, each_image],
                                         batch_size=batch_size,
                                         num_threads=8,
                                         capacity = 3000 + 3*batch_size,
                                         min_after_dequeue = 1000)
    elif mode == 'test':
        _index, _class, _pretrain_class, _positive_norm_class_attr, _negative_norm_class_attr, \
            _class_word2vec, _class_glove, _class_wordnet, _image_feat, _recon_image_feat, _image \
                = tf.train.batch([each_index, each_class, each_pretrain_class, 
                                  each_positive_norm_class_attr, 
                                  each_negative_norm_class_attr,
                                  each_class_word2vec, each_class_glove, each_class_wordnet,
                                  each_image_feat, each_recon_image_feat, each_image],
                                 batch_size=batch_size,
                                 num_threads=1,
                                 capacity = 1000 + 3*batch_size)

    return _index, _class, _pretrain_class, _positive_norm_class_attr, _negative_norm_class_attr, _class_word2vec, _class_glove, _class_wordnet, _image_feat, _recon_image_feat, _image
