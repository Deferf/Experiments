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


def pad_dict(input, d = 8):
  max_length = max([input[k].shape[0] for k in input])
  return {k: torch.cat([input[k], torch.full((max_length - input[k].shape[0], d), float("-inf"))]) for k in input}

def generate_sim_tensor(dict_text, dict_video, order):
  # Input dicts of encoded text and video, keys must match
  # Assumes relationship video-text is one-to-many
  # Hence, there may be multiple captions associated to a video
  # To have an uniform shape across instances we pad with -inf
  padded_text_dict = pad_dict(text_dict)

  # We use stack to group onto a new dimension
  text_tensor = torch.stack([padded_text_dict[k] for k in order], dim = 0)
  # We use cat to group onto an existing dimension
  video_tensor = torch.cat([torch.mean(video_dict[k], dim = 0, keepdim = True) for k in order])
  # We will represent this as a tensor of size (number of instances, max caption of any instance, number of video) 
  sim_tensor = (text_tensor @ video_tensor.T)

  return sim_tensor

def tensor_video_to_text_sim(sim_tensor):
  # Forms a similarity matrix for use with rank at k
  values, _ = torch.max(sim_tensor, dim = 1,keepdim=True)
  return torch.squeeze(values).T

def tensor_video_to_text_sim(sim_tensor, top_k = [1,5,10]):
  # Permute sim_tensor so it represents a sequence of text-video similarity matrices.
  # Then obtain the double argsort to position the rank on the diagonal
  stacked_sim_matrices = sim_tensor.permute(1,0,2)
  first_argsort = torch.argsort(stacked_sim_matrices, dim = -1, descending= True)
  second_argsort = torch.argsort(first_argsort, dim = -1, descending= False)

  # Permute second_argsort so we can extract the diagonal of each text-video similarity matrices
  modified_sa = second_argsort.permute((2,1,0))
  # Extracts ranks i.e diagonals
  ranks = torch.flatten( torch.diagonal(modified_sa))

  # Now we need to extract valid ranks, as some belong to inf padding values
  mask = ~ torch.isinf(torch.flatten(torch.diagonal(stacked_sim_matrices.T)))
  valid_ranks = ranks[mask]
  # A quick dimension check validates our results, there may be other correctness tests pending
  # Such as dot product localization, but that is for other time.
  #assert int(valid_ranks.shape[0]) ==  sum([len(text_dict[k]) for k in text_dict])

  return list_recall(lst, top_k)

def list_recall(lst, top_k):
  # Most of the time we end up with a list (or diagonal) that contains all the ranks
  # We want to obtain results from that
  lst = torch.tensor(lst)
  results = {f"R@{k}" : float(torch.sum(valid_ranks < k) / len(valid_ranks)) for k in top_k}
  results["Median_Rank"] = float(torch.median(diagonal + 1))
  results["Mean_Rank"] = float(torch.mean(diagonal + 1))
  results["Std_Rank"] = float(torch.stf(diagonal + 1))
  return results 