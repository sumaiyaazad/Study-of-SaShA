import pandas as pd
import numpy as np
from config import *

def train_test_split(data, test_size=0.2, train_size=0.8, random_state=0, shuffle=True):
    # split data into train and test
    if shuffle:
        data = data.sample(frac=1, random_state=random_state)

    train_data = data.sample(frac=train_size, random_state=random_state)
    test_data = data.drop(train_data.index)   

    return train_data, test_data


def load_data_ml_1M(split=False):
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

    # train test split
    if split:
        train_data, test_data = train_test_split(data, test_size=(1-TRAIN_SIZE), train_size=TRAIN_SIZE, random_state=0, shuffle=True)

        train_users = users[users['user_id'].isin(train_data['user_id'].unique())]
        train_items = items[items['item_id'].isin(train_data['item_id'].unique())]

        test_users = users[users['user_id'].isin(test_data['user_id'].unique())]
        test_items = items[items['item_id'].isin(test_data['item_id'].unique())]

        # reset index
        train_data = train_data.reset_index(drop=True)
        test_data = test_data.reset_index(drop=True)
        
        train_users = train_users.reset_index(drop=True)
        test_users = test_users.reset_index(drop=True)

        train_items = train_items.reset_index(drop=True)
        test_items = test_items.reset_index(drop=True)

        return (train_data, train_users, train_items), (test_data, test_users, test_items)
    
    else:

        users = users[users['user_id'].isin(data['user_id'].unique())]
        items = items[items['item_id'].isin(data['item_id'].unique())]

        # reset index
        data = data.reset_index(drop=True)
        users = users.reset_index(drop=True)
        items = items.reset_index(drop=True)

        return data, users, items
