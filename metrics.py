from utils import cos_similarity
import numpy as np

def rank_matrix(a, b, axis = 0):
    sm = cos_similarity(a,b)
    l = sm.shape[0]
    return np.array([np.flip(np.argsort(sm[i], axis = axis)) for i in range(l)])

def recall_k(a,b, k = 1, axis = 0, binary = False):
    rm = rank_matrix(a, b, axis = axis)
    recall = []
    for id, row in enumerate(rm):
        items = np.argsort(row.flatten())
        if id in items[:k]:
            recall.append(1)
        else:
            recall.append(0)
    if binary:
        return recall
    else:
        return sum(recall)/len(recall)