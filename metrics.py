from utils import cos_similarity
import numpy as np
import tensorflow as tf
import statistics

def rank_matrix(a, b):
    sm = cos_similarity(a,b)
    return tf.argsort(sm, direction= "DESCENDING")

def recall_metrics(alpha,beta, k = [1,5,10], s = 1):
  # To create the rectangular rank matrix we need some milestones to divide
  length = alpha.shape[0]
  limits = [a for a in range(length,0,-int(length/s))][::-1]
  #print(limits)
  start = 0
  results = {a: 0 for a in k}
  for l in limits:
    end = l
    rm = rank_matrix(alpha[start:end],beta)
    #print(start,end)
    #print(rm)
    for clip in k:
      for i,row in enumerate(rm):
        #print("i in ", i + start)
        if i + start in row[:clip]:
          results[clip] = results[clip] + 1
    start = end
  
  return {"R@" + str(a): [results[a]/length] for a in results}


def rank_at_k(alpha,beta, k = [1,5,10], s = 1):
  # This functions is based on matrix manipulation for speed.
  sim_map = tf.matmul(alpha, tf.transpose(beta))
  return rank_at_k_precomputed(sim_map, k = k)

def rank_at_k_precomputed(sim_map, k = [1,5,10]):
  # This functions is based on matrix manipulation for speed.
  sim_map = tf.convert_to_tensor(sim_map)
  # Obtains first argsort matrix, where first places are at the left
  sim_map_sort = tf.argsort(sim_map,axis=-1,direction='DESCENDING')

  # Obtains second argosrt matrix where diagonals reflect the ranking of the pair itself
  sim_map_sort_2 = tf.argsort(sim_map_sort,axis=-1,direction='ASCENDING')

  length = sim_map.shape[0] # total observations
  diagonal = tf.linalg.tensor_diag_part(sim_map_sort_2).numpy() # diagonal
  results = {clip: sum((diagonal < clip) + 0)  for clip in k} # sum of observations whose values are below k

  metrics = {"R@" + str(a): [results[a]/length] for a in results} # dict composition
  metrics["Median_Rank"] = float(np.median(diagonal + 1))#statistics.median(list(diagonal + 1) ) # diagonal is a list of ranks if you add 1
  metrics["Mean_Rank"] = float(np.mean(diagonal + 1))#statistics.mean(list(diagonal + 1) )
  metrics["Std_Rank"] = float(np.std(diagonal + 1))
  return metrics # ta ta!


def stack_encoded_dict(dictionary, order, processing = lambda x : x):
  stack = []
  aux = []
  for i, key in enumerate(order):
    processed = processing(dictionary[key])
    stack.append(processed)
    for _ in range(len(processed)):
      aux.append(i)
  return torch.cat(stack), torch.tensor([aux]).T

def rank_at_k_precomputed_rectangular(sim_map, k = [1,5,10], aux = None, diag = False):
  length = sim_map.shape[0] # total observations
  # This functions is based on matrix manipulation for speed.
  sim_map = tf.convert_to_tensor(sim_map)
  # Obtains first argsort matrix, where first places are expected to be at the left
  sim_map_sort = tf.argsort(sim_map,axis=-1,direction='DESCENDING')

  # In case of a rectangular matrix an aux vector should be supplied
  if aux != None:
    assert length == aux.shape[0]
    _, diagonal = tf.split(tf.where(sim_map_sort == aux), 2, axis = -1)
    #print(tf.cast(diagonal < 1, tf.int16))
  else:
    # Obtains second argsort matrix where diagonals reflect the ranking of the pair itself
    sim_map_sort_2 = tf.argsort(sim_map_sort,axis=-1,direction='ASCENDING')
    diagonal = tf.linalg.tensor_diag_part(sim_map_sort_2).numpy() # diagonal

  results = {clip: tf.reduce_sum(tf.cast(diagonal < clip, tf.int64))  for clip in k} # sum of observations whose values are below k
  metrics = {"R@" + str(a): [float(results[a]/length)] for a in results} # dict composition
  metrics["Median_Rank"] = float(np.median(diagonal + 1))#statistics.median(list(diagonal + 1) ) # diagonal is a list of ranks if you add 1
  metrics["Mean_Rank"] = float(np.mean(diagonal + 1))#statistics.mean(list(diagonal + 1) )
  metrics["Std_Rank"] = float(np.std(diagonal + 1))
  if diag:
    return metrics, diagonal
  else:
    return metrics