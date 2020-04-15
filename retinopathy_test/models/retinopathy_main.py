# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Runs a ResNet model on the retinopathy dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import retinopathy_test.config as cfg
from absl import app as absl_app #ki: absl is Google's common libraries
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.utils.flags import core as flags_core
from official.utils.logs import logger
import retinopathy_test.models.resnet_model as resnet_model
import retinopathy_test.models.resnet_run_loop as resnet_run_loop

_HEIGHT = 256
_WIDTH = 256
_NUM_CHANNELS = 3
_DEFAULT_IMAGE_BYTES = _HEIGHT * _WIDTH * _NUM_CHANNELS
# The record is the image plus a one-byte label
_RECORD_BYTES = _DEFAULT_IMAGE_BYTES + 1
_NUM_CLASSES = 5
_NUM_DATA_FILES = 1

_NUM_IMAGES = {
    'train': 28101,
    'validation': 7025,
}

DATASET_NAME = 'RETINOPATHY'

###############################################################################
# Data processing
###############################################################################
def get_filenames(is_training, data_dir):
  if is_training:
    #return [os.path.join(data_dir, cfg.Retina_TrainingData)]
    files = [join(data_dir, f) for f in listdir(data_dir) if (isfile(join(data_dir, f)) and "_tr.tfrecords." in f)]  #"_tr."
    # print("[DEBUG] train files: {}".format(files))
    return files
  else:
    #return [os.path.join(data_dir, cfg.Retina_ValidationData)]
    files = [join(data_dir, f) for f in listdir(data_dir) if (isfile(join(data_dir, f)) and "_va.tfrecords." in f)]  #"_va."
    # print("[DEBUG] valid files: {}".format(files))
    return files


def parse_record(example_proto, is_training):
      features = {'image': tf.FixedLenFeature([], tf.string),
                  'label': tf.FixedLenFeature([], tf.int64)}

      parsed_features = tf.parse_single_example(example_proto, features)
      image = tf.decode_raw(parsed_features['image'], tf.float32)
      image = tf.reshape(image, [_HEIGHT, _WIDTH, _NUM_CHANNELS])

      image = preprocess_image(image, is_training)
      image = image / 255.0
      image = image - 0.5

      label = tf.cast(parsed_features['label'], tf.int32)
      return image, label

def preprocess_image(image, is_training):
  """Preprocess a single image of layout [height, width, depth]."""
  if is_training:
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_image_with_crop_or_pad(
        image, _HEIGHT + 8, _WIDTH + 8)

    # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
    image = tf.random_crop(image, [_HEIGHT, _WIDTH, _NUM_CHANNELS])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

    angles = tf.random_uniform([1], -15, 15, dtype=tf.float32, seed=0)
    image = tf.contrib.image.rotate(image, angles * math.pi / 360, interpolation='NEAREST', name=None)
  # Subtract off the mean and divide by the variance of the pixels.
  #image = tf.image.per_image_standardization(image)
  return image


def input_fn(is_training, data_dir, batch_size, num_epochs=1, num_gpus=None):
  """Input_fn using the tf.data input pipeline for retinopathy dataset.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.
    num_gpus: The number of gpus used for training.

  Returns:
    A dataset that can be used for iteration.
  """
  filenames = get_filenames(is_training, data_dir)
  #dataset = tf.data.FixedLengthRecordDataset(filenames, _RECORD_BYTES)
  dataset = tf.data.TFRecordDataset(filenames)

  return resnet_run_loop.process_record_dataset(
      dataset=dataset,
      is_training=is_training,
      batch_size=batch_size,
      shuffle_buffer=_NUM_IMAGES['train']//16, #ki: originally set to 2
      parse_record_fn=parse_record,
      num_epochs=num_epochs,
      num_gpus=num_gpus,
      examples_per_epoch=_NUM_IMAGES['train'] if is_training else None
  )


def get_synth_input_fn():
  return resnet_run_loop.get_synth_input_fn(
      _HEIGHT, _WIDTH, _NUM_CHANNELS, _NUM_CLASSES)

def _get_block_sizes(resnet_size):
  """Retrieve the size of each block_layer in the ResNet model.

  The number of block layers used for the Resnet model varies according
  to the size of the model. This helper grabs the layer set we want, throwing
  an error if a non-standard size has been selected.

  Args:
    resnet_size: The number of convolutional layers needed in the model.

  Returns:
    A list of block sizes to use in building the model.

  Raises:
    KeyError: if invalid resnet_size is received.
  """
  choices = {
      18: [2, 2, 2, 2],
      34: [3, 4, 6, 3],
      50: [3, 4, 6, 3],
      101: [3, 4, 23, 3],
      152: [3, 8, 36, 3],
      200: [3, 24, 36, 3]
  }

  try:
    return choices[resnet_size]
  except KeyError:
    err = ('Could not find layers for selected Resnet size.\n'
           'Size received: {}; sizes allowed: {}.'.format(
               resnet_size, choices.keys()))
    raise ValueError(err)

###############################################################################
# Running the model
###############################################################################
class retinopathyModel(resnet_model.Model):
  """Model class with appropriate defaults for retinopathy data."""

  def __init__(self, resnet_size, data_format='channels_last', num_classes=_NUM_CLASSES,
               resnet_version=resnet_model.DEFAULT_VERSION,
               dtype=resnet_model.DEFAULT_DTYPE):
    """These are the parameters that work for retinopathy data.

    Args:
      resnet_size: The number of convolutional layers needed in the model.
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
      num_classes: The number of output classes needed from the model. This
        enables users to extend the same model to their own datasets.
      resnet_version: Integer representing which version of the ResNet network
      to use. See README for details. Valid values: [1, 2]
      dtype: The TensorFlow dtype to use for calculations.

    Raises:
      ValueError: if invalid resnet_size is chosen
    """
    #if resnet_size % 6 != 2:
    #  raise ValueError('resnet_size must be 6n + 2:', resnet_size)

    bottleneck = True
    final_size = 2048

    super(retinopathyModel, self).__init__(
        resnet_size=resnet_size,
        bottleneck=bottleneck,
        num_classes=num_classes,
        num_filters=64,
        kernel_size=7,
        conv_stride=2,
        first_pool_size=3,
        first_pool_stride=2,
        block_sizes=_get_block_sizes(resnet_size),
        block_strides=[1, 2, 2, 2],
        final_size=final_size,
        resnet_version=resnet_version,
        data_format=data_format,
        dtype=dtype
    )


def retinopathy_model_fn(features, labels, mode, params): #ki: prepares the model to be used in resnet_run_loop.resnet_main
  """Model function for retinopathy."""
  features = tf.reshape(features, [-1, _HEIGHT, _WIDTH, _NUM_CHANNELS])

  learning_rate_fn = resnet_run_loop.learning_rate_with_decay(
      batch_size=params['batch_size'], batch_denom=params['batch_size'],
      num_images=_NUM_IMAGES['train'], boundary_epochs=[30, 60, 80],
      decay_rates=[1.0, 0.1, 0.01, 0.001])

  # We use a weight decay of 0.0001
  weight_decay = 0.0001

  def loss_filter_fn(_):
    return True

  return resnet_run_loop.resnet_model_fn(
      features=features,
      labels=labels,
      mode=mode,
      model_class=retinopathyModel,
      resnet_size=params['resnet_size'],
      weight_decay=weight_decay,
      learning_rate_fn=learning_rate_fn,
      momentum=0.9,
      data_format=params['data_format'],
      resnet_version=params['resnet_version'],
      loss_scale=params['loss_scale'],
      loss_filter_fn=loss_filter_fn,
      dtype=params['dtype']
  )


def define_retinopathy_flags(batch_size=16, train_epochs=10, num_gpus=None):
  resnet_run_loop.define_resnet_flags()
  print("[DEBUG] resnet_flags set")
  flags.adopt_module_key_flags(resnet_run_loop)
  print("[DEBUG] adopt module key flags from resnet_run_loop")
  #flags_core.set_defaults(data_dir='./records/',
                          #model_dir='./retinopathy_model/',
                          #resnet_size='50',
                          #train_epochs=10,
                          #epochs_between_evals=5,
                          #batch_size=1,
                          #export_dir='./retinopathy_serve/')
  flags_core.set_defaults(benchmark_log_dir=cfg.Retina_LocalModels)    #vk: log_dir same as model_dir
  print("[DEBUG] core.set_defaults")
  flags_core.set_defaults(data_dir=cfg.Retina_LocalDataRecords,
                          model_dir=cfg.Retina_LocalModels,
                          resnet_size='50',
                          train_epochs=train_epochs, #10
                          epochs_between_evals=1, #5
                          batch_size=batch_size,
                          num_gpus=num_gpus,
                          export_dir=cfg.Retina_LocalModelsServe,
                          benchmark_logger_type='BenchmarkFileLogger') #vk: create log files



def run_retinopathy(flags_obj):
  """Run ResNet retinopathy training and eval loop.

  Args:
    flags_obj: An object containing parsed flag values.
  """
  graph_zip_path = None

  # verify that set flags are stored correctly in FLAGS. Can be removed #vk
  check_flags_obj = False
  if check_flags_obj:
      for key in flags_obj.flag_values_dict():
          print("{} : {}".format(key, flags_obj[key].value))
        
  # comment it out, as it opts synth dataset #vk
  #input_function = (flags_obj.use_synthetic_data and get_synth_input_fn()
  #                  or input_fn)  
  input_function = (input_fn)
  graph_zip_path = resnet_run_loop.resnet_main(
                                   flags_obj, retinopathy_model_fn, 
                                   input_function, DATASET_NAME,
                                   shape=[_HEIGHT, _WIDTH, _NUM_CHANNELS])

  print(graph_zip_path) # needed to pass this to model.py (??) #vk
  return graph_zip_path


def main(_):
  with logger.benchmark_context(flags.FLAGS):
    run_retinopathy(flags.FLAGS)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  define_retinopathy_flags()
  absl_app.run(main)

