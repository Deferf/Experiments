import tensorflow as tf
from utils import cos_similarity

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

def hard_sampler_full_wrapper(margin):
  def hard_sampler_full(y_true, y_pred):
    # We obtain the similarity matrix and its diagonal
    v,c = tf.split(y_pred, 2, axis = 1)
    S = cos_similarity(v,c)
    St = tf.transpose(S)
    diagonal = tf.linalg.diag_part(S)
    #print(diagonal)
    reshaped = tf.expand_dims(diagonal, axis = 1)#tf.reshape(diagonal,(s[0],1))
    #print(reshaped.shape)
    # Proceed to substract the diagonal to the sims matrix 
    vid_contrast = S - reshaped + margin
    sen_contrast = St - reshaped + margin
    b_loss = tf.maximum(0.0, vid_contrast) + tf.maximum(0.0, sen_contrast)
    b_sum = tf.reduce_sum(b_loss, axis = -1) # Should be mean
    return tf.reduce_mean(b_sum)
  return hard_sampler_full