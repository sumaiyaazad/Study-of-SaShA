import pandas as pd
import time 
from utils.misc import bigskip
from utils.log import Logger


# prediction shift
def prediction_shift(test_data, verbose=False, log=False):
    """
    :param test_data: touple of pandas dataframe (ratings_before_attack, ratings_after_attack)
    
    """
    if verbose:
        print("Calculating prediction shift...")
        start = time.time()

    # if log:
    #     with open(LOG_FILE, 'a') as f:
    #         f.write("{0}\tStarting prediction shift calculation".format(time.time()))

    ratings_before_attack, ratings_after_data = test_data

    num_users = ratings_before_attack.user_id.nunique()
    num_items = ratings_before_attack.item_id.nunique()

    total_prediction_shift = ratings_after_data.ratings.sum() - ratings_before_attack.ratings.sum()
    avg_prediction_shift = total_prediction_shift / (num_users * num_items)

    if verbose:
        print("Prediction shift: {0}".format(avg_prediction_shift))
        print("Time: {0}".format(time.time() - start))

    # if log:
    #     with open(LOG_FILE, 'a') as f:
    #         f.write("{0}\tPrediction shift calculation completed".format(time.time()))

    return avg_prediction_shift

# hit ratio
def hit_ratio(recommendations_filename, target_items, among_firsts, shilling_user_data=None, verbose=False, log=None):
    """
    :param recommendations_filename: filename of the recommendations file
    :param target_items: list of target items
    :param verbose: print the hit ratio
    :param log: log the hit ratio, object of the logger class

    :return: pandas dataframe with the hit ratio for each among_first
    """

    if verbose:
        print("Calculating hit ratio...")
        start = time.time()

    if log is not None:
        log.append("Starting hit ratio calculation")

    recommendations = pd.read_csv(recommendations_filename)
    recommendations.columns = ['user_id', 'item_id', 'rating']


    if shilling_user_data is not None:
        # remove shilling users from recommendations
        recommendations = recommendations[~recommendations['user_id'].isin(shilling_user_data['user_id'].unique().tolist())]


    grouped = recommendations.groupby('user_id', group_keys=True).apply(lambda x: x.sort_values(['rating'], ascending=False))
    grouped.drop('user_id', axis=1, inplace=True)

    tot_users = recommendations.user_id.nunique()

    hit_df = pd.DataFrame(columns=['among_first', 'hit_ratio'])

    for among_first in among_firsts:
        all_hit_counts = grouped.groupby('user_id').head(among_first)['item_id'].value_counts()
        filtered_target_items = [item for item in target_items if item in all_hit_counts.index]
        hits = all_hit_counts[filtered_target_items].sum()
        hit_ratio = hits / (tot_users*len(target_items))
        hit_df.loc[len(hit_df)] = [among_first, hit_ratio]

    
    if verbose:
        print(hit_df)
        print("Time: {0}".format(time.time() - start))

    if log is not None:
        log.append("Hit ratio calculation completed")
    

    return hit_df


def new_hit_ratio(recommendations_filename, target_items, among_firsts, verbose=False, log=None):
    """
    :param recommendations_filename: filename of the recommendations file
    :param target_items: list of target items
    :param verbose: print the hit ratio
    :param log: log the hit ratio, object of the logger class

    :return: pandas dataframe with the hit ratio for each among_first
    """

    if verbose:
        print("Calculating hit ratio...")
        start = time.time()

    if log is not None:
        log.append("Starting hit ratio calculation")

    hit_df = pd.DataFrame(columns=['among_first', 'hit_ratio'])

    for among_first in among_firsts:
        hits = 0
        for item in target_items:

            recommendations = pd.read_csv(recommendations_filename.format(item))
            recommendations.columns = ['user_id', 'item_id', 'rating']
            grouped = recommendations.groupby('user_id', group_keys=True).apply(lambda x: x.sort_values(['rating'], ascending=False))
            grouped.drop('user_id', axis=1, inplace=True)

            tot_users = recommendations.user_id.nunique()
        
            all_hit_counts = grouped.groupby('user_id').head(among_first)['item_id'].value_counts()
            hits += 0 if item not in all_hit_counts.index else all_hit_counts[item]
        
        hit_ratio = hits / (tot_users*len(target_items))
        hit_df.loc[len(hit_df)] = [among_first, hit_ratio]

    
    if verbose:
        print(hit_df)
        print("Time: {0}".format(time.time() - start))

    if log is not None:
        log.append("Hit ratio calculation completed")
    

    return hit_df

def accuracy(recommendations_filename, target_items, verbose=False, log=None):
    """
    :param recommendations_filename: filename of the recommendations file
    :param target_items: list of target items
    :param verbose: print the hit ratio
    :param log: log the hit ratio, object of the logger class

    :return: pandas dataframe with the hit ratio for each among_first
    """

    if verbose:
        print("Calculating accuracy...")
        start = time.time()

    if log is not None:
        log.append("Starting accuracy calculation")

    recommendations = pd.read_csv(recommendations_filename)
    recommendations.columns = ['user_id', 'item_id', 'rating']
    grouped = recommendations.groupby('user_id', group_keys=True).apply(lambda x: x.sort_values(['rating'], ascending=False))
    grouped.drop('user_id', axis=1, inplace=True)

    tot_users = recommendations.user_id.nunique()

    all_hit_counts = grouped.groupby('user_id').head(1)['item_id'].value_counts()
    filtered_target_items = [item for item in target_items if item in all_hit_counts.index]
    hits = all_hit_counts[filtered_target_items].sum()
    accuracy = hits / tot_users
    
    if verbose:
        print("Accuracy: {0}".format(accuracy))
        print("Time: {0}".format(time.time() - start))

    if log is not None:
        log.append("Accuracy calculation completed")
    

    return accuracy

def shilling_profile_detection_accuracy(shilling_profiles, detected_profiles):
    """
    :param shilling_profiles: pandas df of shilling profiles
    :param detected_profiles: pandas df of detected profiles

    :return: accuracy of the detection
    """

    shilling_profiles = set(shilling_profiles['user_id'].unique().tolist())
    detected_profiles = set(detected_profiles['user_id'].unique().tolist())

    accuracy = len(shilling_profiles.intersection(detected_profiles)) / len(shilling_profiles)

    return accuracy
