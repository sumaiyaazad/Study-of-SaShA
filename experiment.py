
####################################################################################################
# experiment flow:
# 
# choose dataset
#   load data
#   list most popular items
#   list most unpopular items
#
#   choose similarity measure
#       generate pre-attack similarities
#
#       choose recommender system
#           generate pre-attack recommendations
#           calculate pre-attack hit ratio
#
#           choose attack
#               generate attack profiles
#               generate post-attack similarities
#               generate post-attack recommendations
#               calculate post-attack hit ratio
# 
#               choose detection method
#                   generate detected attack profiles
#                   generate post-detection similarities
#                   generate post-detection recommendations
#                   calculate post-detection hit ratio
#                   calculate detection accuracy
####################################################################################################



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

        # choose similarity measure --------------------------------------------------------------------------------------------------------
        pre_attack_similarities_dir = dirname + 'pre_attack_similarities/'
        os.makedirs(pre_attack_similarities_dir, exist_ok=True)

        post_attack_similarities_dir = dirname + 'post_attack_similarities/'
        os.makedirs(post_attack_similarities_dir, exist_ok=True)

        post_detection_similarities_dir = dirname + 'post_detection_similarities/'
        os.makedirs(post_detection_similarities_dir, exist_ok=True)

        for similarity in SIMILARITY_MEASURES:

            if similarity == 'cosine':
                similarity_filename = 'cosine_similarity_{}.csv'.format(dataset)
            elif similarity == 'pearson':
                similarity_filename = 'pearson_similarity_{}.csv'.format(dataset)
            elif similarity == 'adjusted_cosine':
                similarity_filename = 'adjusted_cosine_similarity_{}.csv'.format(dataset)
            else:
                if args.log:
                    log.append('similarity measure {} not found'.format(similarity))
                    log.abort()

                noti.balloon_tip('SAShA Detection', 'Similarity measure {} not found. Experiment aborted.'.format(similarity))
                raise ValueError('Similarity measure not found.')
            
            print('Proceeding with similarity measure {}'.format(similarity))

            if args.log:
                log.append('Proceeding with similarity measure {}'.format(similarity))

            # choose recommender system -----------------------------------------------------------------------------------------------------
            for rs_model in RS_MODELS:
                if rs_model == 'ibcf':
                    from recommender_systems.memory_based.item_based_CF import ItemBasedCF as RS
                    rs = RS(train, similarity_filename, similarity=similarity, notification_level=0, log=log if args.log else None)


                elif rs_model == 'ubcf':
                    from recommender_systems.memory_based.user_based_CF import UserBasedCF as RS
                    rs = RS(train, similarity_filename, similarity=similarity, notification_level=0, log=log if args.log else None)


                elif rs_model == 'mfcf':
                    from recommender_systems.model_based.matrix_factorization_CF import MatrixFactorizationCF as RS
                    mfcf_train_data, mfcf_train_user, mfcf_train_item = convert_to_matrix(train_data, train_users, train_items)

                    rs = RS(mfcf_train_data, mfcf_train_user, mfcf_train_item, K=K, alpha=ALPHA, beta=BETA, iterations=MAX_ITER, notification_level=0, log=log if args.log else None)


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


