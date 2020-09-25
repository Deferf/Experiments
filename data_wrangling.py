from datetime import datetime
import tensorflow as tf
import numpy as np

# Functions for TFRecord exporting
def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example_2_Tensors(feature0, feature1):
  """
  Creates a tf.train.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.train.Example-compatible
  # data type.
  feature = {
      'feature0': _bytes_feature(tf.io.serialize_tensor(feature0)),
      'feature1': _bytes_feature(tf.io.serialize_tensor(feature1)),
      #'feature1': _float_feature(feature1)
      #'feature1': _bytes_feature(feature1), --> To serialize text
  }

  # Create a Features message using tf.train.Example.

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()

# Functions for TFRecord importing
def parse_function_2_Tensors(example_proto):
    feature_description = {
    'feature0': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'feature1': tf.io.FixedLenFeature([], tf.string, default_value='')
    }

    # Parse the input `tf.train.Example` proto using the dictionary above.
    pair = tf.io.parse_single_example(example_proto, feature_description)
    pair['feature0'] = tf.io.parse_tensor(pair['feature0'], tf.float32)
    pair['feature1'] = tf.io.parse_tensor(pair['feature1'], tf.float32)

    return pair['feature0'], pair['feature1']


def TF_Record_Writer_2_Tensors(filename, serializer_function, features_array):
  f0 = features_array[0]
  f1 = features_array[1]
  n_examples = f0.shape[0]
  timestamp = datetime.now().strftime(" %y_%m_%d %H:%M:%S")
  f = filename + timestamp + ".tfrecord"
  with tf.io.TFRecordWriter(f) as writer:
    for i in range(n_examples):
      example = serializer_function(f0[i], f1[i])
      writer.write(example)
  print("Successfully written on " + f)

def TF_Record_Writer_2_Tensors_Iterative_Batch(filename, serializer_function, features_array,mapping_function):
  # filename = path
  # serializer_function = will transform features into tfrecord compatible format
  # features_array = expects an array with features to be stored into the format [[f0,f1], [f1]]
  # mapping functions = a funciton to call per feature, same length as features array
  assert len(features_array) == len(mapping_function)
  f0 = features_array[0] # visual names -> sentence
  f1 = features_array[1] # literal sentences -> captions
  timestamp = datetime.now().strftime(" %y_%m_%d %H:%M:%S")
  f = filename + timestamp + ".tfrecord"
  v_curr = ""
  v_emb = 0
  with tf.io.TFRecordWriter(f) as writer:
    for v, v_name in enumerate(f0):
      if not v_name == v_curr:
        # current video
        v_curr = v_name
        v_emb = mapping_function[0](v_name)
      # Example a tuple with the concatenation of visual and text embeddigns, and text embeddings
      # Remember to make them float32
      # Concatenateee!
      s_emb_c = mapping_function[1](f1[v])
      v_emb_c = tf.convert_to_tensor(np.concatenate([v_emb, s_emb_c]), dtype= tf.float32)
      s_emb = tf.convert_to_tensor(s_emb_c, dtype= tf.float32)
      example = serializer_function(v_emb_c, s_emb)
      writer.write(example)
      print("'\r{0}".format(v), end='')
  print("Successfully written on " + f)

