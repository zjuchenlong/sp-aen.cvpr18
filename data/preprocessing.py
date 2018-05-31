import os
import cfg
import scipy.io
import tensorflow as tf
from CUB_extra import save_cub_extra_data, read_cub_extra_data
from SUN import save_sun_data, read_sun_data
from AWA2 import save_awa2_data, read_awa2_data
from aPY import save_apy_data, read_apy_data

FLAGS = tf.app.flags.FLAGS


def _read_split_txt(splitpath, splittype):
    def _remove_n(data_list):
        ret_list = []
        for item_i in data_list:
            ret_list.append(item_i.split('\n')[0])
        return ret_list

    def _each_split(datapath):
        with open(datapath) as f:
            each_split_data = f.readlines()
        return _remove_n(each_split_data)

    trainvalclasses_path = os.path.join(splitpath, 'trainvalclasses.txt')
    testclasses_path = os.path.join(splitpath, 'testclasses.txt')

    if splittype == 'type0':
        trainclasses_path = trainvalclasses_path
        valclasses_path = trainvalclasses_path
    elif splittype == 'type1':
        trainclasses_path = os.path.join(splitpath, 'trainclasses1.txt')
        valclasses_path = os.path.join(splitpath, 'valclasses1.txt')
    elif splittype == 'type2':
        trainclasses_path = os.path.join(splitpath, 'trainclasses2.txt')
        valclasses_path = os.path.join(splitpath, 'valclasses2.txt')
    elif splittype == 'type3':
        trainclasses_path = os.path.join(splitpath, 'trainclasses3.txt')
        valclasses_path = os.path.join(splitpath, 'valclasses3.txt')
    else:
        raise NotImplementedError

    trainvalclasses = _each_split(trainvalclasses_path)
    testclasses = _each_split(testclasses_path)
    trainclasses = _each_split(trainclasses_path)
    valclasses = _each_split(valclasses_path)
    return trainvalclasses, testclasses, trainclasses, valclasses


def _get_class_image_mean(class2images, index2class, net_inputsize, image_path):
    print "get class image mean"
    class_image_mean = {}
    for class_name in index2class.itervalues():
        print class_name
        class_images = np.zeros((len(class2images[class_name]), net_inputsize, net_inputsize, 3), dtype=np.float32)
        for i, each_image in enumerate(class2images[class_name]):
            class_images[i] = _crop_image(os.path.join(image_path, each_image), 
                                          target_height=net_inputsize,
                                          target_width=net_inputsize)

        class_image_mean[class_name] = class_images
        break

    return class_image_mean


def save_data(save_tfrecords=False, extra_data=False, save_type='tf'):
    if FLAGS.dataset == 'cub':
        dataset_info = save_cub_extra_data(save_tfrecords=save_tfrecords, save_type=save_type)
    elif FLAGS.dataset == 'sun':
        dataset_info = save_sun_data(save_tfrecords=save_tfrecords, save_type=save_type)
    elif FLAGS.dataset == 'awa2':
        dataset_info = save_awa2_data(save_tfrecords=save_tfrecords, save_type=save_type)
    elif FLAGS.dataset == 'apy':
        dataset_info = save_apy_data(save_tfrecords=save_tfrecords, save_type=save_type)
    else:
        raise NotImplementedError
    return dataset_info
  

def read_data(datapath, batch_size, net_inputsize, mode):
    assert os.path.exists(datapath), 'There is no %s'%(datapath)
    filename_queue = tf.train.string_input_producer([datapath])

    # get imagenet mean, but in BGR not RGB
    matfile = scipy.io.loadmat(os.path.join(cfg.PRETRAIN_MODEL, 'ilsvrc_2012_mean.mat'))
    image_mean = matfile['image_mean']
    topleft = ((image_mean.shape[0] - net_inputsize)/2, (image_mean.shape[1] - net_inputsize)/2)
    crop_image_mean = image_mean[topleft[0]:topleft[0]+net_inputsize, topleft[1]:topleft[1]+net_inputsize]

    if FLAGS.dataset == 'cub':
        _index, _class, _pretrain_class, _positive_norm_class_attr, _negative_norm_class_attr, _image_feat, _comp_image_feat, _image \
            = read_cub_data(filename_queue, batch_size, crop_image_mean, mode)
    elif FLAGS.dataset == 'sun':
        _index, _class, _pretrain_class, _positive_norm_class_attr, _negative_norm_class_attr, _image_feat, _comp_image_feat, _image \
            = read_sun_data(filename_queue, batch_size, crop_image_mean, mode)
    elif FLAGS.dataset == 'awa2':
        _index, _class, _pretrain_class, _positive_norm_class_attr, _negative_norm_class_attr, _image_feat, _comp_image_feat, _image \
            = read_awa2_data(filename_queue, batch_size, crop_image_mean, mode)
    elif FLAGS.dataset == 'apy':
        _index, _class, _pretrain_class, _positive_norm_class_attr, _negative_norm_class_attr, _image_feat, _comp_image_feat, _image \
            = read_apy_data(filename_queue, batch_size, crop_image_mean, mode)
    else:
        raise NotImplementedError

    return _index, _class, _pretrain_class, _positive_norm_class_attr, _negative_norm_class_attr, _image_feat, _comp_image_feat, _image
