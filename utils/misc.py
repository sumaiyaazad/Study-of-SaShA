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
