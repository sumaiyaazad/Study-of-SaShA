import pandas as pd
import numpy as np
from config import *

def train_test_split(data, test_size=0.2, train_size=0.8, random_state=0, shuffle=True):
    # split data into train and test
    if shuffle:
        data = data.sample(frac=1, random_state=random_state)

    train_data = data.sample(frac=train_size, random_state=0)
    test_data = data.drop(train_data.index).sample(frac=test_size, random_state=0)
    

    return train_data, test_data


def load_data_ml_1M():
    # Load ratings data
    data = pd.read_csv('data/ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')
    users = pd.read_csv('data/ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
    items = pd.read_csv('data/ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')


    # Rename columns
    data.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    items.columns = ['item_id', 'title', 'genres']
    users.columns = ['user_id', 'gender', 'age', 'occupation', 'zip-code']

    # Drop timestamp column
    data = data.drop('timestamp', axis=1)

    # randomly sample 25% of the data for faster experimentation
    # [ref: https://link.springer.com/chapter/10.1007/978-3-030-49461-2_18]
    data = data.sample(frac=SAMPLE_FRAC, random_state=0)

    # to avoid cold start drop users with less than 5 ratings and items with less than 5 ratings 
    # [ref: https://link.springer.com/chapter/10.1007/978-3-030-49461-2_18]

    data = data.groupby('user_id').filter(lambda x: len(x) >= COLD_START_THRESHOLD)
    data = data.groupby('item_id').filter(lambda x: len(x) >= COLD_START_THRESHOLD)

    users = users[users['user_id'].isin(data['user_id'].unique())]
    items = items[items['item_id'].isin(data['item_id'].unique())]

    return data, users, items



def convert_to_matrix(data, users, items):
    """
    Convert the data to a matrix 
    for matrix factorization
    """

    users_dict = {k: v for v, k in enumerate(users['user_id'].tolist())}
    items_dict = {k: v for v, k in enumerate(items['item_id'].tolist())}

    matrix = np.zeros((len(users), len(items)))
    for row in data.itertuples():
        matrix[users_dict[row[1]], items_dict[row[2]]] = row[3]
    return matrix, users_dict, items_dict
