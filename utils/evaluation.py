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
def hit_ratio(recommendations_filename, hits_filename, target_items, amoung_firsts, verbose=False, log=None):
    """
    :param recommendations_filename: filename of the recommendations file
    :param target_items: list of target items
    :param verbose: print the hit ratio
    :param log: log the hit ratio, object of the logger class
    """

    if verbose:
        print("Calculating hit ratio...")
        start = time.time()

    if log is not None:
        log.append("Starting hit ratio calculation")

    recommendations = pd.read_csv(recommendations_filename, header=None, names=['user_id', 'item_id', 'rating'])
    grouped = recommendations.groupby('user_id').apply(lambda x: x.sort_values(['rating'], ascending=False))
    grouped.drop('user_id', axis=1, inplace=True)

    tot_users = recommendations.user_id.nunique()

    for amoung_first in amoung_firsts:
        all_hit_counts = grouped.groupby('user_id').head(amoung_first)['item_id'].value_counts()
        hits = all_hit_counts[target_items].sum()
        hit_ratio = hits / tot_users*len(target_items)
    
