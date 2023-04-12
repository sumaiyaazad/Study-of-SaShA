import pandas as pd
import time 
from config import LOG_FILE


# prediction shift
def prediction_shift(test_data, verbose=False, log=False):
    """
    :param test_data: touple of pandas dataframe (ratings_before_attack, ratings_after_attack)
    
    """
    if verbose:
        print("Calculating prediction shift...")
        start = time.time()

    if log:
        with open(LOG_FILE, 'a') as f:
            f.write("{0}\tStarting prediction shift calculation".format(time.time()))

    ratings_before_attack, ratings_after_data = test_data

    num_users = ratings_before_attack.user_id.nunique()
    num_items = ratings_before_attack.item_id.nunique()

    total_prediction_shift = ratings_after_data.ratings.sum() - ratings_before_attack.ratings.sum()
    avg_prediction_shift = total_prediction_shift / (num_users * num_items)

    if verbose:
        print("Prediction shift: {0}".format(avg_prediction_shift))
        print("Time: {0}".format(time.time() - start))

    if log:
        with open(LOG_FILE, 'a') as f:
            f.write("{0}\tPrediction shift calculation completed".format(time.time()))

    return avg_prediction_shift

# hit ratio
def hit_ratio(recommendations, target_items, verbose=False, log=False):
    """
    :param recommendations: dictionary of recommended items for each user in the test set
    :param target_items: list of target items
    """

    if verbose:
        print("Calculating hit ratio...")
        start = time.time()

    if log:
        with open(LOG_FILE, 'a') as f:
            f.write("{0}\tStarting hit ratio calculation".format(time.time()))

    hits = 0
    for item in target_items:
        for recs in recommendations.values():
            if item in recs:
                hits += 1

    if verbose:
        print("Hit ratio: {0}".format(hits / len(recommendations) * len(target_items)))
        print("Time: {0}".format(time.time() - start))

    if log:
        with open(LOG_FILE, 'a') as f:
            f.write("{0}\tHit ratio calculation completed".format(time.time()))

    return hits / len(recommendations)*len(target_items)
