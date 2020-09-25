import tensorflow as tf
from utils import cos_similarity

def naive_max_ranking_roll_wrapper(margin = 0.2, n = 0):
    def naive_max_ranking_roll(y_true,y_p):
        y_neg = tf.roll(y_p, shift=1, axis = 0)
        vidp, senp = tf.split(y_p, 2, axis = 1)
        vidn, senn = tf.split(y_neg, 2, axis = 1)
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

# Needs revision, what is in first principles a hard datapoint?
# What does the loss function does when n = batch size?
def hard_sampler_wrapper(margin = 0.2, n = 1):
  def hard_sampler(y_true, y_pred):
    # We obtain the similarity matrix and its diagonal
    v,c = tf.split(y_pred, 2, axis = 1)
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
  def proxy_sampler(y_true, y_pred):
    # We obtain the similarity matrix of sentences and its diagonal
    C = cos_similarity(y_true,y_true)
    # Here C is negative to obtain the position of the most dissimilar sentences
    indices_p = tf.math.top_k(-C, k = n)[1]
    # Now we start with v and c similarity
    v,c = tf.split(y_pred, 2, axis = 1)
    S = cos_similarity(v,c)
    St = tf.transpose(S)
    diagonal = tf.linalg.diag_part(S)
    #print(diagonal)
    reshaped = tf.expand_dims(diagonal, axis = 1)#tf.reshape(diagonal,(s[0],1))
    # Extract from the v-c sim matrix the positions obtained by the proxy
    values_s = tf.gather(S,indices_p, batch_dims=1)
    values_st = tf.gather(St,indices_p, batch_dims=1)

    vid_contrast = values_s - reshaped #+ margin
    sen_contrast = values_st - reshaped #+ margin

    b_loss = tf.maximum(0.0, vid_contrast + margin) + tf.maximum(0.0, sen_contrast + margin)
    b_sum = tf.reduce_sum(b_loss, axis = -1) # Should be mean
    return tf.reduce_mean(b_sum)
  return proxy_sampler

def iterative_loss_wrapper(margin = 0.2, n = 1,):
    def iterative_loss(y_true,y_p):
        # v and c are the respective video and captions tensors from the batch, shape (b, [repr. dim.])
        # n is the number of samples per pair
        # t type of pairs to sample

        # For each datapoint find the 3 most distant
        v,c = tf.split(y_p, 2, axis = 1)

        # D is supposed to be b x b
        D = cos_similarity(v,c)
        print(D, tf.shape(D))
        # vp_cp is just the repetition for n pairs
        s = tf.shape(D)
        # Convert the matrix into vector
        diagonal = tf.linalg.diag_part(D)
        reshaped = tf.reshape(diagonal,(s[0],1))
        # Repeat each item into n columns
        repeated = tf.tile(reshaped, [1,n])
        # Again convert matrix into vector
        vp_cp = tf.reshape(repeated, [n * s[0]])

        # Change dialect, rows represent video
        # Here we eliminate the diagonal to only choose negatives by ranking the respective rows and columns
        rows = D - tf.linalg.diag(tf.linalg.diag_part(D)) - tf.eye(s[0])
        # Columns represent captions
        columns = tf.transpose(rows)

        values_r = tf.math.top_k(rows, k = n)[0]
        #values_r = -values_r
        s_r = tf.shape(values_r)

        values_c = tf.math.top_k(columns, k = n)[0]
        #values_c = -values_c
        s_c = tf.shape(values_c)
        
        vp_cn = tf.reshape(values_r, [n * s_r[0]])
        vn_cp = tf.reshape(values_c, [n * s_c[0]])

        loss = tf.maximum(0.0, margin + vp_cn - vp_cp) + tf.maximum(0.0, margin + vn_cp - vp_cp)
        loss = tf.reduce_mean(loss) + 1e-12
        return loss

    return iterative_loss

# Dude this does work but I dont know why :(())
def proxy_sampler_wrapper_2(margin = 0.2, n = 1):
  def loss(y_true, y_pred):
    proxy = cos_similarity(y_true, y_true)
    v,c = tf.split(y_pred, 2, axis = 1)
    S = cos_similarity(v,c)
    St = tf.transpose(S)
    s = tf.shape(S)
    diagonal = tf.linalg.diag_part(S)
    #print(diagonal)
    reshaped = tf.expand_dims(diagonal, axis = 1)#tf.reshape(diagonal,(s[0],1))
    print(reshaped.shape)
    # Proceed to substract the diagonal to the sims matrix 
    vid_contrast = S - reshaped + margin
    sen_contrast = St - reshaped + margin
    b_loss = tf.maximum(0.0, vid_contrast) + tf.maximum(0.0, sen_contrast)
    b_sum = tf.reduce_sum(b_loss, axis = -1) # Should be mean
    return tf.reduce_mean(b_sum)

    return proxy
  return loss