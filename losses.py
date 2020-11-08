import tensorflow as tf
from utils import cos_similarity

def naive_max_ranking_roll_wrapper(margin = 0.2, n = 0):
    def naive_max_ranking_roll(y_none,y_pred):
        vidp, senp, y_true = tf.split(y_pred, [2048,2048,768], axis = 1)
        vidn = tf.roll(vidp, shift=1, axis = 0)
        senn = tf.roll(senp, shift=1, axis = 0)
        # It's called Roll because treats its neighbor as negative
        vp_sn = cos_similarity(vidp, senn)
        d_vp_sn = tf.linalg.diag_part(vp_sn)
        r_vp_sn = tf.expand_dims(d_vp_sn, axis = 1)
        vp_sp = cos_similarity(vidp, senp)
        d_vp_sp = tf.linalg.diag_part(vp_sp)
        r_vp_sp = tf.expand_dims(d_vp_sp, axis = 1)
        vn_sp = cos_similarity(vidn, senp)
        d_vn_sp = tf.linalg.diag_part(vn_sp)
        r_vn_sp = tf.expand_dims(d_vn_sp, axis = 1)
        # Max ranking loss
        loss = tf.maximum(0.0, margin + r_vp_sn - r_vp_sp) + tf.maximum(0.0, margin + r_vn_sp - r_vp_sp)
        loss = tf.reduce_mean(loss) + 1e-12
        return loss
    return naive_max_ranking_roll

def hard_sampler_full_wrapper(margin = 0.2, n = 0):
  def hard_sampler_full(y_none, y_pred):
    # We obtain the similarity matrix and its diagonal
    v,c, y_true = tf.split(y_pred, [2048,2048,768], axis = 1)
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

# Needs revision, what is in first principles a hard datapoint?
# What does the loss function does when n = batch size?
def hard_sampler_wrapper(margin = 0.2, n = 1):
  def hard_sampler(y_none, y_pred):
    # We obtain the similarity matrix and its diagonal
    v,c, y_true = tf.split(y_pred, [2048,2048,768], axis = 1)
    S = cos_similarity(v,c)
    St = tf.transpose(S)
    diagonal = tf.linalg.diag_part(S)
    #print(diagonal)
    reshaped = tf.expand_dims(diagonal, axis = 1)#tf.reshape(diagonal,(s[0],1))

    # Set the diagonal to -1 so it is never chosen as the next best candidate
    S = tf.linalg.set_diag(S, -tf.math.pow(diagonal,0))
    St = tf.linalg.set_diag(St, -tf.math.pow(diagonal,0))
    #print(reshaped.shape)
    # Proceed to substract the diagonal to the sims matrix
    values_s = tf.math.top_k(S, k = n)[0]
    vid_contrast = values_s - reshaped #+ margin

    values_st = tf.math.top_k(St, k = n)[0]
    sen_contrast = values_st - reshaped #+ margin
    
    b_loss = tf.maximum(0.0, vid_contrast + margin) + tf.maximum(0.0, sen_contrast + margin)
    b_sum = tf.reduce_sum(b_loss, axis = -1) # Should be mean
    return tf.reduce_mean(b_sum)
  return hard_sampler

def proxy_sampler_wrapper(margin = 0.2, n = 1):
  def proxy_sampler_3(y_none, y_pred):
    v, c, y_true = tf.split(y_pred, [2048,2048,768], axis = 1)
    C = cos_similarity(y_true,y_true)
    values_p, indices_p = tf.math.top_k(-C, k = n)
    
    S = cos_similarity(v,c)
    St = tf.transpose(S)

    diagonal = tf.linalg.diag_part(S)

    reshaped = tf.expand_dims(diagonal, axis = 1)
    
    values_s = tf.gather(S,indices_p, batch_dims=1)
    values_st = tf.gather(St,indices_p, batch_dims=1)

    vid_contrast = values_s - reshaped + margin
    sen_contrast = values_st - reshaped + margin
    
    b_loss = tf.maximum(0.0, vid_contrast) + tf.maximum(0.0, sen_contrast)
    
    b_sum = tf.reduce_sum(b_loss, axis = -1)
    
    return b_sum
  return proxy_sampler_3