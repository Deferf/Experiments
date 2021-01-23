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
  # Obtains first argsort matrix, where first places are at the left
  sim_map_sort = tf.argsort(sim_map,axis=-1,direction='DESCENDING')

  # Obtains second argosrt matrix where firstplaces are at diagonals
  sim_map_sort_2 = tf.argsort(sim_map_sort,axis=-1,direction='ASCENDING')

  length = alpha.shape[0] # total observations
  diagonal = tf.linalg.tensor_diag_part(sim_map_sort_2).numpy() # diagonal
  results = {clip: sum((diagonal < clip) + 0)  for clip in k} # sum of observations whose values are below k

  metrics = {"R@" + str(a): [results[a]/length] for a in results} # dict composition
  metrics["MedRank"] = statistics.median(list(diagonal + 1) ) # diagonal is a list of ranks if you add 1
  return metrics # ta ta!