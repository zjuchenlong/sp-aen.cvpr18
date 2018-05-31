
# Zero-Shot Visual Recognition using Semantics-Preserving Adversarial Embedding Networks

This repository contains the code for the following paper:

* Long Chen, Hanwang Zhang, Jun Xiao, Wei Liu, Shih-Fu Chang, *Zero-Shot Visual Recognition using Semantics-Preserving Adversarial Embedding Networks*. In CVPR, 2018. [[PDF](https://arxiv.org/pdf/1712.01928.pdf)]


Note: This repository is adaptived from [DeepSim](https://github.com/shijx12/DeepSim), [tensorflow-resnet](https://github.com/ry/tensorflow-resnet), Thanks a lot to [Jiaxin Shi](https://github.com/shijx12) from Tsinghua University, and [Yongqin Xian](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly/)  to release his dataset split and code.

## Requirements and Dependencies
* Python 2.7 ([Anaconda](https://www.continuum.io/downloads) recommended)
* [TensorFlow](https://www.tensorflow.org/install/) 
* [Caffe](http://caffe.berkeleyvision.org/) (Optional, In my original experiment, I load pretrained ImageNet CNN model weigths in caffemodel format and save it to npy format for tensorflow. But we can direct download the npy format CNN pretrained weights)

## Get Started
### Dataset and Pretrain Model Download
* Download the dataset ([CUB](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)/[AWA](https://cvml.ist.ac.at/AwA2/)/[aPY](http://vision.cs.uiuc.edu/attributes/)/[SUN](http://vision.cs.princeton.edu/projects/2010/SUN/)) and dataset split file ([proposed_split/standard_split](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly/)), and save it into directory /SP-AEN-HOME/data/, and change the corresponding path of dataset and split file in **/SP-AEN-HOME/cfg.py**.
* Download pretrained CNN model [alexnet](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet), [caffenet](https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet), [resnet](https://github.com/ry/tensorflow-resnet), and save the pretrained model in /SP-AEN-HOME/model/pretrained/imagenet. And download pretrained model [caffenet/fc6](https://lmb.informatik.uni-freiburg.de/resources/binaries/cvpr2016_invert_alexnet/fc6.zip) and save to /SP-AEN-HOME/model/pretrained/generator/caffenet/fc6.  Use **/SP-AEN-HOME/model/caffe2tf.py** to convert the caffemodel format weights to npy format. 

###  Data Preprocessing
Extract the resnet-101 CNN featuer and preprocessing the dataset directly use:
```shell
python /SP-AEN-HOME/convert_data.py -dataset cub (for CUB)
python /SP-AEN-HOME/convert_data.py -dataset apy (for aPY)
python /SP-AEN-HOME/convert_data.py -dataset awa2 (for AWA2)
python /SP-AEN-HOME/convert_data.py -dataset sun (for SUN)
```

### Train: 
dataset can select from ['cub', 'apy', 'awa2', 'sun'], for dataset CUB:
```shell
step1:
python run.py --dataset=cub --mode=trainf --summary_path=.. --alpha_map_rank=.. --alpha_rank_dis=.. --alpha_rank_gen=..
step2:
python run.py --dataset=cub --mode=traine --retrain_model=True --train_checkpoint=.. --summary_path=..
```

### Citiation
If you find this code useful, please cite the following paper:
```
@inproceedings{chen2018zero,
  title={Zero-Shot Visual Recognition using Semantics-Preserving Adversarial Embedding Networks},
  author={Chen, Long and Zhang, Hanwang and Xiao, Jun and Liu, Wei and Chang, Shih-Fu},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2018}
}
```