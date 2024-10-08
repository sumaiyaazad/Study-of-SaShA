import numpy as np
import math as m

def handleNan(x, y):
    """
    remove nan values from x and y
    """
    if len(x) or len(y) == 0:
        return x, y

    x_nan = np.isnan(x)
    y_nan = np.isnan(y)
    index = np.logical_or(x_nan, y_nan)
    x = x[~index]
    y = y[~index]
    return x, y

def get_common_ratings(x, y):
    """
    get common ratings from x and y
    """
    common = set(x).intersection(set(y))  
    x = [x[i] for i in common]
    y = [y[i] for i in common]
    return x, y
    

# -------------------------------- similarity measures ----------------------------------
# cosine similarity
def cosine_similarity(x, y):

    if type(x) == dict:
        x, y = get_common_ratings(x, y)
    else:
        x, y = handleNan(x, y)

    if np.sum(x) == 0 or np.sum(y) == 0:
        return 0

    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

# jaccard similarity
def jaccard_similarity(x, y):
    x, y = handleNan(x, y)
    return np.sum(np.minimum(x, y)) / np.sum(np.maximum(x, y))

# dice similarity
def dice_similarity(x, y):
    x, y = handleNan(x, y)
    return 2 * np.sum(np.minimum(x, y)) / (np.sum(x) + np.sum(y))

# tanimoto similarity
def tanimoto_similarity(x, y):
    x, y = handleNan(x, y)
    return np.sum(np.minimum(x, y)) / (np.sum(x) + np.sum(y) - np.sum(np.minimum(x, y)))


# adjusted cosine similarity
def adjusted_cosine_similarity(x, y):
    
    if type(x) == dict:
        x, y = get_common_ratings(x, y)
    else:
        x, y = handleNan(x, y)

    if np.sum(x) == 0 or np.sum(y) == 0:
        return 0
    
    x = x - np.mean(x)
    y = y - np.mean(y)
    
    if np.sum(x) == 0 or np.sum(y) == 0:
        return 0
    
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

# -------------------------------- distance measures -------------------------------------
# euclidean distance
def euclidean_distance(x, y):
    x, y = handleNan(x, y)
    return np.sqrt(np.sum(np.square(x - y)))

# manhattan distance
def manhattan_distance(x, y):
    x, y = handleNan(x, y)
    return np.sum(np.absolute(x - y))

# minkowski distance
def minkowski_distance(x, y, p):
    x, y = handleNan(x, y)
    return np.power(np.sum(np.power(np.absolute(x - y), p)), 1/p)

# chebyshev distance
def chebyshev_distance(x, y):
    x, y = handleNan(x, y)
    return np.max(np.absolute(x - y))

# hamming distance
def hamming_distance(x, y):
    x, y = handleNan(x, y)
    return np.sum(x != y)


# -------------------------------- correlation measures ---------------------------------
# pearson correlation
def pearson_correlation(x, y):

    """
    x_mean and y_mean are the mean of x and y for all users
    """

    if type(x) == dict:
        x, y = get_common_ratings(x, y)
    else:
        x, y = handleNan(x, y)

    if np.sum(x) == 0 or np.sum(y) == 0:
        return 0
    
    x = x - np.mean(x)
    y = y - np.mean(y)
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

# spearman correlation
def spearman_correlation(x, y):
    x, y = handleNan(x, y)
    x = np.argsort(x)
    y = np.argsort(y)
    return pearson_correlation(x, y)
