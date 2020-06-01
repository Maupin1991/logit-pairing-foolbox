import sys
import os
import numpy as np

import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2

import foolbox
from foolbox import zoo

def create():
    tf.enable_eager_execution()

    class Model(object):
        def __init__(self):
            # load model
            self.input_ = tf.keras.layers.Input(shape=(64, 64, 3), dtype=tf.float32)
            model_fn_two_args = get_model('resnet_v2_50', 1001)
            self.model_fn = lambda x: model_fn_two_args(self._input_, is_training=False)


        def __call__(self, inputs):
            sess = tf.get_default_session()
            preprocessed = _normalize(inputs)
            self.logits = model_fn(preprocessed)[:, 1:]
            sess.run(self.logits, feed_dict={self.input_: preprocessed})


    with tf.get_default_graph().as_default():
        # load pretrained weights
        weights_path = zoo.fetch_weights(
            'http://download.tensorflow.org/models/adversarial_logit_pairing/imagenet64_alp025_2018_06_26.ckpt.tar.gz',
            unzip=True
        )
        checkpoint = os.path.join(weights_path, 'imagenet64_alp025_2018_06_26.ckpt')
        model = Model()
        # load pretrained weights into model
        variables_to_restore = tf.contrib.framework.get_variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        sess = tf.Session().__enter__()


        saver.restore(sess, checkpoint)

    # create foolbox model
    fmodel = foolbox.models.TensorFlowModel(model, bounds=(0, 255))
    
    return fmodel

def _normalize(image):
  """Rescale image to [-1, 1] range."""
  return tf.multiply(tf.subtract(image, 0.5), 2.0)

def get_model(model_name, num_classes):
  """Returns function which creates model.

  Args:
    model_name: Name of the model.
    num_classes: Number of classes.

  Raises:
    ValueError: If model_name is invalid.

  Returns:
    Function, which creates model when called.
  """
  if model_name.startswith('resnet'):
    def resnet_model(images, is_training, reuse=tf.AUTO_REUSE):
      with tf.contrib.framework.arg_scope(resnet_v2.resnet_arg_scope()):
        resnet_fn = resnet_v2.resnet_v2_50
        return resnet_fn
    return resnet_model
  else:
    raise ValueError('Invalid model: %s' % model_name)
