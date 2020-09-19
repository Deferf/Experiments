import tensorflow as tf

def naive_max_ranking_roll(y_true,y_p, margin = 0.2):
    y_neg = tf.roll(y_p, shift=1, axis = 0)
    vidp, senp = tf.split(y_p, 2, axis = 1)
    vidn, senn = tf.split(y_neg, 2, axis = 1)

    vp_sn = cos_similarity(vidp, senn)
    vp_sp = cos_similarity(vidp, senp)
    vn_sp = cos_similarity(vidn, senp)
    # Max ranking loss
    loss = tf.maximum(0.0, margin + vp_sn - vp_sp) + tf.maximum(0.0, margin + vn_sp - vp_sp)
    loss = tf.reduce_mean(loss) + 1e-12
    return loss

def cos_similarity(a, b):
    a_norm = tf.math.l2_normalize(a, axis = 1)
    b_norm = tf.math.l2_normalize(b, axis = 1)
    return tf.matmul(a_norm,tf.transpose(b_norm))