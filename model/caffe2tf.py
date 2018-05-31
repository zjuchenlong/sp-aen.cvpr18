try:
    import caffe
except ImportError:
    print "!!!! Caffe import error"
    pass
import numpy as np
from collections import OrderedDict
import tensorflow as tf
import os

def save_model(deploy, model, savename):
    net = caffe.Net(deploy, model, caffe.TEST)

    params_ = OrderedDict()
    for name, layer in net.params.iteritems():
        params_[name] = {}
        assert len(layer) == 2

        if name.startswith('conv'):
            params_[name]['weights'] = layer[0].data.transpose(2, 3, 1, 0).copy()
            params_[name]['biases'] = layer[1].data.copy()
        elif name.startswith('fc'):
            params_[name]['weights'] = layer[0].data.T.copy()
            params_[name]['biases'] = layer[1].data.copy()
        elif name.startswith('deconv'):
            params_[name]['weights'] = layer[0].data.transpose(2, 3, 1, 0).copy()
            params_[name]['biases'] = layer[1].data.copy()
        elif name.startswith('defc'):
            params_[name]['weights'] = layer[0].data.T.copy()
            params_[name]['biases'] = layer[1].data.copy()
        else:
            raise NotImplementedError

    np.savez(savename, **params_)
    print "caffemodel parameters had been saved"

def load_model(model_path):
    model_data = np.load(model_path).items()
    
    params_ = OrderedDict()
    for layer in model_data:
       layer_name, layer_params = layer
       params_[layer_name] = {'weights': layer_params.item()['weights'], 
                              'biases': layer_params.item()['biases']}
    return params_

def _test():
    deploy = './pretrained/imagenet/caffenet/caffenet.prototxt'
    model = './pretrained/imagenet/caffenet/caffenet.caffemodel'
    model_dir = os.path.dirname(model)
    model_name = os.path.basename(model).split('.')[0]
    model_path = os.path.join(model_dir, model_name + '_params')
    if not os.path.exists(model_path + '.npz'):
        save_model(deploy, model, model_path)
    model_data = load_model(model_path + '.npz')
    
    for name, weights  in model_data.iteritems():
        print('Layer: %s, weights shape: %s, biases shape: %s'%(name, 
                                                                weights['weights'].shape,
                                                                weights['biases'].shape))
        
if __name__ == "__main__":
    _test()
