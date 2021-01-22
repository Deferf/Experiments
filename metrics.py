from utils import cos_similarity
import numpy as np
import tensorflow as tf

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
  # To create the rectangular rank matrix we need some milestones to divide
  length = alpha.shape[0] #height
  limits = [a+1 for a in range(0,length,int(length/s))] # creates upper limits
  start = 0
  results = {a: 0 for a in k}
  rank_list = []
  for l in limits:
    end = l
    rm = rank_matrix(alpha[start:end],beta) # a rectangle size limit size x width, with rows argsorted

    for i,row in enumerate(rm):
      rank_list.append(int(row[start + i]) + 1) # identify the sorting position and add 1 to convert to rank
      for clip in k:
        if start + i in row[:clip]:
          results[clip] = results[clip] + 1

    start = end
  
  metrics = {"R@" + str(a): [results[a]/length] for a in results}
  metrics["MedRank"] = statistics.median(rank_list)
  return metrics