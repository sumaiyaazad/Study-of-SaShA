
import pandas as pd

def load_data_ml_1M_ratings(file_name = 'data/ml-1m/ratings.dat'):
    # Load data
    data = pd.read_csv(file_name, sep='::', header=None, engine='python', encoding='latin-1')

    # Rename columns
    data.columns = ['user_id', 'item_id', 'rating', 'timestamp']

    # Drop timestamp column
    data = data.drop('timestamp', axis=1)

    return data

def load_data_ml_1M_items(file_name = 'data/ml-1m/movies.dat'):
    # Load data
    data = pd.read_csv(file_name, sep='::', header=None, engine='python', encoding='latin-1')

    # Rename columns
    data.columns = ['item_id', 'title', 'genres']

    return data

def load_data_ml_100k_ratings(file_name = 'data/ml-100k/u.data', sep='\t'):
    # Load data
    data = pd.read_csv(file_name, sep=sep, header=None, engine='python', encoding='latin-1')

    # Rename columns
    data.columns = ['user_id', 'item_id', 'rating', 'timestamp']

    # Drop timestamp column
    data = data.drop('timestamp', axis=1)

    return data

def load_data_ml_1M_users(file_name = 'data/ml-1m/users.dat'):
    # Load data
    data = pd.read_csv(file_name, sep='::', header=None, engine='python', encoding='latin-1')

    # Rename columns
    data.columns = ['user_id', 'gender', 'age', 'occupation', 'zip-code']

    return data


def train_test_split(data, test_size=0.2, train_size=0.8, random_state=0, shuffle=True):
    # split data into train and test
    if shuffle:
        data = data.sample(frac=1, random_state=random_state)

    train_data = data.sample(frac=train_size, random_state=0)
    test_data = data.drop(train_data.index).sample(frac=test_size, random_state=0)
    

    return train_data, test_data
