import tensorflow as tf
from data.preprocessing import save_data

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', '', 'cub|apy|awa2|sun, which dataset to do experiment')
tf.app.flags.DEFINE_string('encoder', 'resnet', 'which type of CNN feature')
tf.app.flags.DEFINE_string('recon_encoder', 'caffenet', 'types of CNN network for comparator')
tf.app.flags.DEFINE_integer('resnet_layer', 101, 'layer for resnet encoder')
tf.app.flags.DEFINE_integer('single_class_num', '-1', 'which class to store')

# save_type in ['tf'|'h5']
def main(_):
    dataset_info = save_data(save_tfrecords=True, save_type='h5')

if __name__=='__main__':
    tf.app.run()
