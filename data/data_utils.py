import os
import cfg
import cv2
import numpy as np
import skimage.io
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def get_test_crop(each_image, image_path, net_inputsize):
    origin_image = crop_image(os.path.join(image_path, each_image['path']), 
                                target_height=cfg.SAVE_IMAGE_HEIGHT,
                                target_width=cfg.SAVE_IMAGE_WIDTH)
    
    return cv2.resize(origin_image, (net_inputsize, net_inputsize))

def get_test_crop_with_path(each_image_path, image_path, net_inputsize):
    origin_image = crop_image(os.path.join(image_path, each_image_path), 
                                target_height=cfg.SAVE_IMAGE_HEIGHT,
                                target_width=cfg.SAVE_IMAGE_WIDTH)
    
    return cv2.resize(origin_image, (net_inputsize, net_inputsize))

def get_ten_extra_crop(each_image, image_path, net_inputsize):
    origin_image = crop_image(os.path.join(image_path, each_image['path']), 
                                target_height=cfg.SAVE_IMAGE_HEIGHT,
                                target_width=cfg.SAVE_IMAGE_WIDTH)

    top_point = int((cfg.SAVE_IMAGE_HEIGHT - net_inputsize)/2)
    left_point = int((cfg.SAVE_IMAGE_WIDTH - net_inputsize)/2) 
    image1 = origin_image[0:net_inputsize, 0:net_inputsize]
    image2 = origin_image[-net_inputsize:, -net_inputsize:]
    image3 = origin_image[-net_inputsize:, 0:net_inputsize]
    image4 = origin_image[0:net_inputsize, -net_inputsize:]
    image5 = origin_image[top_point: top_point+net_inputsize, left_point: left_point+net_inputsize]
    image1_flip = image1[:, ::-1, :]
    image2_flip = image2[:, ::-1, :]
    image3_flip = image3[:, ::-1, :]
    image4_flip = image4[:, ::-1, :]
    image5_flip = image5[:, ::-1, :]
    resize_origin_image = cv2.resize(origin_image, (net_inputsize, net_inputsize))

    return np.array([image1, image1_flip, image2, image2_flip, image3, image3_flip, image4, image4_flip, image5, image5_flip, resize_origin_image])

def get_ten_crop(each_image, image_path, net_inputsize):
    origin_image = crop_image(os.path.join(image_path, each_image['path']), 
                                target_height=cfg.SAVE_IMAGE_HEIGHT,
                                target_width=cfg.SAVE_IMAGE_WIDTH)

    top_point = int((cfg.SAVE_IMAGE_HEIGHT - net_inputsize)/2)
    left_point = int((cfg.SAVE_IMAGE_WIDTH - net_inputsize)/2) 
    image1 = origin_image[0:net_inputsize, 0:net_inputsize]
    image2 = origin_image[-net_inputsize:, -net_inputsize:]
    image3 = origin_image[-net_inputsize:, 0:net_inputsize]
    image4 = origin_image[0:net_inputsize, -net_inputsize:]
    image5 = origin_image[top_point: top_point+net_inputsize, left_point: left_point+net_inputsize]
    image1_flip = image1[:, ::-1, :]
    image2_flip = image2[:, ::-1, :]
    image3_flip = image3[:, ::-1, :]
    image4_flip = image4[:, ::-1, :]
    image5_flip = image5[:, ::-1, :]

    return np.array([image1, image1_flip, image2, image2_flip, image3, image3_flip, image4, image4_flip, image5, image5_flip])

def get_ten_crop_with_path(each_image_path, image_path, net_inputsize):
    origin_image = crop_image(os.path.join(image_path, each_image_path), 
                                target_height=cfg.SAVE_IMAGE_HEIGHT,
                                target_width=cfg.SAVE_IMAGE_WIDTH)

    top_point = int((cfg.SAVE_IMAGE_HEIGHT - net_inputsize)/2)
    left_point = int((cfg.SAVE_IMAGE_WIDTH - net_inputsize)/2) 
    image1 = origin_image[0:net_inputsize, 0:net_inputsize]
    image2 = origin_image[-net_inputsize:, -net_inputsize:]
    image3 = origin_image[-net_inputsize:, 0:net_inputsize]
    image4 = origin_image[0:net_inputsize, -net_inputsize:]
    image5 = origin_image[top_point: top_point+net_inputsize, left_point: left_point+net_inputsize]
    image1_flip = image1[:, ::-1, :]
    image2_flip = image2[:, ::-1, :]
    image3_flip = image3[:, ::-1, :]
    image4_flip = image4[:, ::-1, :]
    image5_flip = image5[:, ::-1, :]

    return np.array([image1, image1_flip, image2, image2_flip, image3, image3_flip, image4, image4_flip, image5, image5_flip])


def convert_mat_to_list(mat, dataset, convert_type='simple'):
    ret_list = []
    if dataset == 'cub':
        for i in mat:
            ret_list.append(os.path.basename(i[0][0]))
    elif dataset == 'sun':
        if convert_type == 'simple':
            for i in mat:
                ret_list.append(str("/".join(i[0][0].split('/')[8:])))
        elif convert_type == 'origin':
            for i in mat:
                ret_list.append(str(i[0][0]))
                # import pdb; pdb.set_trace()
    elif dataset == 'awa2':
        if convert_type == 'origin':
            for i in mat:
                ret_list.append(str("/".join(i[0][0].split('/'))))
        elif convert_type == 'simple':
            for i in mat:
                ret_list.append(str("/".join(i[0][0].split('/')[-2:])))
    elif dataset == 'apy':
        if convert_type == 'simple':
            for i in mat:
                i_part = i[0][0].split('/')
                if len(i_part) == 12: # VOC image
                    ret_list.append(str("/".join(i[0][0].split('/')[8:])))
                elif len(i_part) == 9: #aYahoo image
                    ret_list.append(str("/".join(i[0][0].split('/')[-2:])))
                else:
                    raise NotImplementedError
    else:
        raise NotImplementedError

    return ret_list

def checkfile(datapath):
    assert os.path.exists(datapath), 'This is no file %s'%(datapath)
    return datapath

def checkdir(datapath):
    if not os.path.exists(datapath):
        os.makedirs(datapath)
    return datapath

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def norm_feat(data):
    """
    e.g. CUB class attr size: 200 x 312
    """
    assert data.ndim == 2
    data_len = np.linalg.norm(data, axis=1)
    data_len += 1e-8
    norm_data = data / data_len[:, None]
    return norm_data

def crop_image(x, target_height=227, target_width=227):
    # skimage.img_as_float convert image np.ndarray into float type, with range (0, 1)
    # image = skimage.img_as_float(skimage.io.imread(x)).astype(np.float32)
    image = skimage.io.imread(x)

    if image.ndim == 2:
        image = image[:,:,np.newaxis][:,:,[0,0,0]]  # convert the gray image to rgb image
    elif image.ndim == 4:
        image = image[0]
    elif image.ndim == 3 and image.shape[-1] == 4:
        image = image[:, :, :3]

    height, width, rgb = image.shape

    if width == height:
        resized_image = cv2.resize(image, (target_width,target_height))

    elif height < width:
        resized_image = cv2.resize(image, (int(width * float(target_height)/height), target_height))
        cropping_length = int((resized_image.shape[1] - target_width) / 2)
        resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

    else:
        resized_image = cv2.resize(image, (target_width, int(height * float(target_width) / width)))
        cropping_length = int((resized_image.shape[0] - target_height) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

    return cv2.resize(resized_image, (target_width, target_height))