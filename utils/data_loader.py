import pandas as pd
import numpy as np
from config import *
from ast import literal_eval

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

        train_users = users.copy()
        train_items = items.copy()

        test_users = users.copy()
        test_items = items.copy()

        # train_users = users[users['user_id'].isin(train_data['user_id'].unique())]
        # train_items = items[items['item_id'].isin(train_data['item_id'].unique())]

        # test_users = users[users['user_id'].isin(test_data['user_id'].unique())]
        # test_items = items[items['item_id'].isin(test_data['item_id'].unique())]

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


def load_data_dummy():

    data = pd.read_csv('data/dummy/ratings.csv', header=None)
    users = pd.read_csv('data/dummy/users.csv', header=None)
    items = pd.read_csv('data/dummy/items.csv', header=None)

    data.columns = ['user_id', 'item_id', 'rating']
    users.columns = ['user_id']
    items.columns = ['item_id']

    return (data, users, items), None


def load_data_yahoo_movies(split=False):

    # Load ratings data
    data = pd.read_csv('data/yahoo_movies/ratings.csv', header=None)
    data.columns = ['user_id', 'item_id', 'rating']


    # randomly sample 25% of the data for faster experimentation
    # [ref: https://link.springer.com/chapter/10.1007/978-3-030-49461-2_18]
    data = data.sample(frac=SAMPLE_FRAC, random_state=0)

    # to avoid cold start drop users with less than 5 ratings and items with less than 5 ratings 
    # [ref: https://link.springer.com/chapter/10.1007/978-3-030-49461-2_18]
    # avoid cold start
    data = data.groupby('user_id').filter(lambda x: len(x) >= 5)
    data = data.groupby('item_id').filter(lambda x: len(x) >= 5)

    # reset user and item ids
    data['user_index'] = data['user_id']
    data['item_index'] = data['item_id']

    users = pd.DataFrame(data['user_index'].unique(), columns=['user_id'])
    items = pd.DataFrame(data['item_index'].unique(), columns=['item_id'])

    users.reset_index(inplace=True)
    items.reset_index(inplace=True)

    users.columns = ['user_id', 'user_index']
    items.columns = ['item_id', 'item_index']

    for r in users.itertuples():
        data.loc[data['user_index'] == r.user_index, 'user_id'] = r.user_id

    for r in items.itertuples():
        data.loc[data['item_index'] == r.item_index, 'item_id'] = r.item_id

    data.drop(['user_index', 'item_index'], axis=1, inplace=True)

    # train test split
    if split:
        train_data, test_data = train_test_split(data, test_size=(1-TRAIN_SIZE), train_size=TRAIN_SIZE, random_state=0, shuffle=True)        

        train_users = users.copy()
        train_items = items.copy()

        test_users = users.copy()
        test_items = items.copy()

        # reset index
        train_data.reset_index(drop=True)
        test_data.reset_index(drop=True)
        
        train_users.reset_index(drop=True)
        test_users.reset_index(drop=True)

        train_items.reset_index(drop=True)
        test_items.reset_index(drop=True)

        return (train_data, train_users, train_items), (test_data, test_users, test_items)
    
    else:
        return data, users, items
    

def load_kg_yahoo_movies(items, feature_type='ontological'):
    """
    Load KG data
    params:
        items: items dataframe
    returns:
        (kg, features, items)
        kg: kg dataframe
        features: features dataframe
        items: items dataframe (input)
    """

    selected_features = pd.read_csv('data/yahoo_movies/selected_features.csv')
    selected_features = literal_eval(selected_features.loc[selected_features['type'] == feature_type]['features'].to_list()[0])

    kg = pd.read_csv('data/yahoo_movies/df_map.csv')
    kg.columns = ['feature', 'item_id', 'item_index_discard', 'value']

    kg['feature'] = kg['feature'].astype('int64')
    kg['item_id'] = kg['item_id'].astype('int64')

    print('kg shape: ', kg.shape)

    # remove features not in selected_features
    kg = kg[kg['feature'].isin(selected_features)]

    print('kg shape after removing features not in selected_features: ', kg.shape)

    # remove kg entries of items not in items
    kg = kg[kg['item_id'].isin(items['item_index'])]

    print('kg shape after removing kg entries of items not in items: ', kg.shape)

    kg.drop(['item_index_discard'], axis=1, inplace=True)
    kg.drop(['value'], axis=1, inplace=True)

    kg['feature_id'] = kg['feature']

    features = pd.DataFrame(kg['feature'].unique(), columns=['feature_id'])

    features.reset_index(inplace=True)
    features.columns = ['feature_id', 'feature']

    for r in items.itertuples():
        kg.loc[kg['item_id'] == r.item_index, 'item_id'] = r.item_id

    for r in features.itertuples():
        kg.loc[kg['feature'] == r.feature, 'feature_id'] = r.feature_id

    kg.drop(['feature'], axis=1, inplace=True)

    kg.reset_index(drop=True)
    return (kg, features, items)