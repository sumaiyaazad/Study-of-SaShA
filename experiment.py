import argparse
import pandas as pd
from config import *
from utils.data_loader import *
import os
from utils.misc import *
import utils.notification as noti
from utils.log import Logger

def main():

    # ------------------------------------------- define experiment environment -------------------------------------------
    print('------------------------------------------- Experiment Environment -------------------------------------------')
    print('Dataset: ', DATASETS)
    print('Recommender system: ', RS_MODELS)
    print('Similarity measure: ', SIMILARITY_MEASURES)
    print('Attack: ', ATTACKS)
    print('Attack impact evaluation metrics: ', EVALUATIONS)
    print('Detection: ', DETECTIONS)
    print('Experiment start time: ', now())

    # experiment result directory
    num_of_dirs = len([name for name in os.listdir(OUTDIR) if os.path.isdir(os.path.join(OUTDIR, name)) and name.startswith('experiment_results_')])

    dirname = OUTDIR + 'experiment_results_' + str(num_of_dirs) + '/'
    print('Experiment result directory: ', dirname)
    os.makedirs(dirname, exist_ok=True)
    bigskip()

    if args.log:
        LOG_FILE = dirname + 'log.txt'

        log = Logger(LOG_FILE)
        log.append('experiment started')

    for dataset in DATASETS:
        # load data -----------------------------------------------------------------------------------------------------------------------
        if dataset == 'ml-1m':
            train, test = load_data_ml_1M(split=True)
        else:
            if args.log:
                log.append('dataset {} not found'.format(dataset))
                log.abort()

            noti.balloon_tip('SAShA Detection', 'Dataset {} not found. Experiment aborted.'.format(dataset))
            raise ValueError('Dataset not found.')
        
        if args.log:
            log.append('dataset {} loaded'.format(dataset))
        
        train_data, train_users, train_items = train

        # sort items by average rating ----------------------------------------------------------------------------------------------------
        items_sorted = train_data.groupby('item_id')['rating'].mean().to_frame()
        items_sorted.reset_index(inplace=True)
        items_sorted = items_sorted.rename(columns = {'index':'item_id', 'rating':'avg_rating'})
        items_sorted = items_sorted.sort_values(by=['avg_rating'], ascending=False)
        items_sorted.reset_index(inplace=True)
        items_sorted = items_sorted.drop(columns=['index'])

        # list most popular items
        popular_items = items_sorted.head(NUM_TARGET_ITEMS)
        popular_items.to_csv(dirname + 'popular_items_{}.csv'.format(dataset), index=False)
        print('generated {} popular items for dataset {}'.format(NUM_TARGET_ITEMS, dataset))
        if args.log:
            log.append('generated {} popular items for dataset {}. Saved in file {}'.format(NUM_TARGET_ITEMS, dataset, dirname + 'popular_items_{}.csv'.format(dataset)))

        # list most unpopular items; to be used lates as target items of push attacks
        unpopular_items = items_sorted.tail(NUM_TARGET_ITEMS)
        unpopular_items.to_csv(dirname + 'unpopular_items_{}.csv'.format(dataset), index=False)
        print('generated {} unpopular items for dataset {}'.format(NUM_TARGET_ITEMS, dataset))
        if args.log:
            log.append('generated {} unpopular items for dataset {}. Saved in file {}'.format(NUM_TARGET_ITEMS, dataset, dirname + 'unpopular_items_{}.csv'.format(dataset)))


        # choose recommender system -------------------------------------------------------------------------------------------------------
        for rs_model in RS_MODELS:
            for similarity in SIMILARITY_MEASURES:
                if rs_model == 'ibcf':
                    from recommender_systems.memory_based import ItemBasedCF as RS
                    rs = RS(train, test, similarity_measure=similarity, k=IKNN)
                elif rs_model == 'ubcf':
                    from recommender_systems.memory_based import UserBasedCF as RS
                    rs = RS(train, test, similarity_measure=similarity, k=UKNN)
                elif rs_model == 'SVD':
                    from recommender_systems.model_based import SVD as RS
                    rs = RS(train, test, k=K)
                else:
                    if args.log:
                        log.append('recommender system {} not found'.format(rs_model))
                        log.abort()
                    noti.balloon_tip('SAShA Detection', 'Recommender system {} not found. Experiment aborted.'.format(rs_model))
                    raise ValueError('Recommender system not found.')
                


# i can optimize by generating simmilarity matrix only once for each dataset and saving them in a seperate file prehand will save time for multiple experiments



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', type=bool, default=True, help='verbose mode')
    parser.add_argument('--noti_level', type=int, default=0, help='notification level, 0: no notification, 1: only at the end, 2: at verbose mode')
    parser.add_argument('--log', type=bool, default=True, help='log mode')

    args = parser.parse_args()
    main()
