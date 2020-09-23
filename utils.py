import tensorflow as tf

def cos_similarity(a, b):
    a_norm = tf.math.l2_normalize(a, axis = 1)
    b_norm = tf.math.l2_normalize(b, axis = 1)
    return tf.matmul(a_norm,tf.transpose(b_norm))