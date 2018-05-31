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

def _read_class(ps_attr_data):
    """
    Already preprocess name with _
    """
    allclasses_names = ps_attr_data['allclasses_names']
    allclasses_names_list = convert_mat_to_list(allclasses_names, 'awa2', convert_type='origin')
    class2index = {}
    index2class = {}
    for i, each_class in enumerate(allclasses_names_list, 1):
        class2index[each_class] = i
        index2class[i] = each_class

    return class2index, index2class

# def _read_images(image_path, index2class):
#     index2image = {}
#     image2index = {}
#     all_images = {}
#     class2images = {}

#     total_image_index = 0
#     for i in xrange(1, cfg.AWA2_CLASS_NUM+1):
#         class_name = index2class[i]
#         class_image_path = os.path.join(image_path, class_name)

#         class_image_list = os.listdir(class_image_path)
#         sorted(class_image_list)

#         class2images[class_name] = []
#         for each_i, each_image in enumerate(class_image_list, 1):
#             image_index = each_i + total_image_index
#             index2image[image_index] = each_image
#             image2index[each_image] = image_index
#             class2images[class_name].append(each_image)
#             all_images[each_image] = {'index': image_index, 'class':class_name}
#         total_image_index = total_image_index + len(class_image_list)

#     return class2images, all_images, index2image, image2index


def _read_images(ps_image_filenames_list):
    index2image = {}
    image2index = {}
    all_images = {}
    class2images = {}

    for i, each_image in tqdm(enumerate(ps_image_filenames_list, 1)):
        image2index[each_image] = i
        index2image[i] = each_image

        # each_image_part = each_image.split("/")
        # each_image_class = "_".join(each_image_part[1:-1])
        each_image_class = each_image.split("/")[0]
        if not each_image_class in class2images:
            class2images[each_image_class] = []
        class2images[each_image_class].append(each_image)
        all_images[each_image] = {'index': i, 'class': each_image_class}

    return class2images, all_images, index2image, image2index


def convert_to_tfrecords(writer, imageinfo, batch_enc_image_feat, batch_recon_image_feat, batch_image, encoder_net_inputsize):
    for each_enc_image_feat, each_recon_image_feat, each_image in zip(batch_enc_image_feat, batch_recon_image_feat, batch_image):
        each_image_raw = each_image.reshape((-1, encoder_net_inputsize*encoder_net_inputsize*3)).tobytes()
        each_recon_image_feat = each_recon_image_feat.reshape((4096,))
        example = tf.train.Example(features=tf.train.Features(feature={
                'index': int64_feature([imageinfo['index']]),
                'class': int64_feature([imageinfo['class']]),
                'pretrain_class': int64_feature([imageinfo['pretrain_class']]),
                'class_attr': float_feature(imageinfo['class_attr']),
                'image_feat':float_feature(each_enc_image_feat),
                'recon_image_feat':float_feature(each_recon_image_feat),
                'image':bytes_feature(each_image_raw)
            }))
        # print example
        writer.write(example.SerializeToString())

def save_awa2_data(save_tfrecords=False, save_type='tf'):
    attr_dim = cfg.AWA2_ATT_DIM
    class_num = cfg.AWA2_CLASS_NUM
    image_num = cfg.AWA2_IMAGE_NUM
    ss_split_path = os.path.join(cfg.SS_SPLIT_PATH, 'AWA2')
    ps_split_path = os.path.join(cfg.PS_SPLIT_PATH, 'AWA2')
    save_data_path = checkdir(os.path.join(cfg.PREPROCESSED_DATA_PATH, 'AWA2'))
    image_path = os.path.join(cfg.AWA2_PATH, 'JPEGImages')

    data_path = cfg.AWA2_PATH

    # ss_attr_data = io.loadmat(checkfile(os.path.join(ss_split_path, 'att_splits.mat')))
    ps_attr_data = io.loadmat(checkfile(os.path.join(ps_split_path, 'att_splits.mat')))
    ps_image_filenames = io.loadmat(checkfile(os.path.join(ps_split_path, 'ps_image_files.mat')))['image_files']
    ps_image_filenames_list = convert_mat_to_list(ps_image_filenames, 'awa2')

    class2index, index2class = _read_class(ps_attr_data)
    # class2images, allimages, index2image, image2index =  _read_images(image_path, index2class)
    class2images, allimages, index2image, image2index = _read_images(ps_image_filenames_list)

    class_attr = ps_attr_data['att'].T

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
            assert each_image['path'] == ps_image_path
            each_image['class'] = class2index[allimages[each_image['path']]['class']] # start from 1
            each_image['class_attr'] = class_attr[each_image['class']-1]
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
                    checkdir(os.path.join(save_data_path, '%s_%s'%(FLAGS.encoder, FLAGS.recon_encoder)))
                    tfrecords_save_path = os.path.join(save_data_path, '%s_%s'%(FLAGS.encoder, FLAGS.recon_encoder), each_save_split+'.tfrecords')
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
                    checkdir(os.path.join(save_data_path, '%s_%s'%(FLAGS.encoder, FLAGS.recon_encoder)))
                    tfrecords_save_path = os.path.join(save_data_path, '%s_%s'%(FLAGS.encoder, FLAGS.recon_encoder), each_save_split+'.tfrecords')
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

            elif save_type == 'h5':
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
                            h5_feat_file['class_attr'][each_image_i] = each_image['class_attr']
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
                            h5_feat_file['class_attr'][each_image_i] = each_image['class_attr']
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
            'class_attr':class_attr, 
            'trainval_class_num': trainval_class_number,
            'train_class_num': train_class_number,
            'val_class_num': val_class_number,
            'test_seen_class_num': test_seen_class_number,
            'test_unseen_class_num': test_unseen_class_number,
            'ps_trainval_images': ps_trainval_images,
            'ps_test_unseen_images': ps_test_unseen_images}

def read_awa2_data(filename_queue, batch_size, image_mean, mode):
    attr_dim = cfg.AWA2_ATT_DIM

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
                'class_attr': tf.FixedLenFeature([attr_dim], tf.float32),
                'image_feat': tf.FixedLenFeature([feat_dim], tf.float32),
                'recon_image_feat': tf.FixedLenFeature([4096], tf.float32),
                'image': tf.FixedLenFeature([encoder_net_inputsize*encoder_net_inputsize*3], tf.string)
            }
        )

    each_index = tf.cast(features['index'], tf.int32)
    each_class = tf.cast(features['class'], tf.int32)
    each_pretrain_class = tf.cast(features['pretrain_class'], tf.int32)
    each_class_attr = features['class_attr']
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
        _index, _class, _pretrain_class, _class_attr, _image_feat, _recon_image_feat, _image \
                = tf.train.shuffle_batch([each_index, each_class, each_pretrain_class, 
                                          each_class_attr, each_image_feat, each_recon_image_feat, each_image],
                                         batch_size=batch_size,
                                         num_threads=8,
                                         capacity = 2000 + 3*batch_size,
                                         min_after_dequeue = 1000)
    elif mode == 'test':
        _index, _class, _pretrain_class, _class_attr, _image_feat, _recon_image_feat, _image \
                = tf.train.batch([each_index, each_class, each_pretrain_class, 
                                  each_class_attr, each_image_feat, each_recon_image_feat, each_image],
                                 batch_size=batch_size,
                                 num_threads=3,
                                 capacity = 2000 + 3*batch_size)

    return _index, _class, _pretrain_class, _class_attr, None, _image_feat, _recon_image_feat, _image