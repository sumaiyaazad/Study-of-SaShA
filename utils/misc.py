import time
import numpy as np

def now():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def bigskip():
    print('\n\n\n')

def dummy():
    pass

def convert_to_matrix(data, users, items):
    """
    Convert the data to a matrix 
    for matrix factorization
    """

    # reset the index
    users = users.reset_index(drop=True)
    items = items.reset_index(drop=True)
    data = data.reset_index(drop=True)

    users_dict = {k: v for v, k in enumerate(users['user_id'].tolist())}
    items_dict = {k: v for v, k in enumerate(items['item_id'].tolist())}

    matrix = np.zeros((len(users), len(items)))
    for row in data.itertuples():
        matrix[users_dict[row[1]], items_dict[row[2]]] = row[3]
    return matrix, users_dict, items_dict


def get_item_feature_matrix(kg, items, features):
    """
    Get the item feature matrix
    if value is 1, then the item has the feature
    if value is 0, then the item does not have the feature
    """
    nitems = len(items)
    nfeatures = len(features)

    item_feature_matrix = np.zeros((nitems, nfeatures))

    for row in kg.itertuples():
        item_feature_matrix[row[1], row[2]] = 1.
    return item_feature_matrix