from datetime import datetime

import tensorflow as tf
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