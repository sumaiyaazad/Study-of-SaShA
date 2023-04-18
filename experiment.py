import argparse
import pandas as pd
from config import *
from utils.data_loader import *
from utils.similarity_measures import *
import os
from utils.misc import *
import utils.notification as noti
from utils.log import Logger
from utils.sendmail import sendmail, sendmailwithfile

def generateRecommendations(train, rs_model, similarity, similarities_dir, recommendation_filename, log):
    """
    Generate recommendations for all users in the training set
    param train: training set
    param rs_model: recommender system model
    param similarity: similarity measure
    param similarities_dir: directory to store similarity files
    param recommendation_filename: filename of storing recommendation result
    param log: object of Logger class
    """

    if similarity == 'cosine':
        similarity_function = cosine_similarity
    elif similarity == 'pearson':
        similarity_function = pearson_correlation
    elif similarity == 'jaccard':
        similarity_function = jaccard_similarity
    elif similarity == 'adjusted_cosine':
        similarity_function = adjusted_cosine_similarity
    else:
        if args.log:
            log.append('similarity measure {} not found'.format(similarity))
            log.abort()
        noti.balloon_tip('SAShA Detection', 'Similarity measure {} not found. Experiment aborted.'.format(similarity))
        raise ValueError('Similarity measure not found.')

    if rs_model == 'ibcf':
        from recommender_systems.memory_based.item_based_CF import ItemBasedCF as RS
        similarity_filename = similarities_dir + 'item_item_' + similarity + '.csv'

        rs = RS(train, similarity_filename, similarity=similarity_function, notification_level=0, log=log if args.log else None)
        rs.getRecommendationsForAllUsers(n_neighbors=IKNN, verbose=True, output_filename=recommendation_filename, sep=',', top_n=TOP_N)

    elif rs_model == 'ubcf':
        similarity_filename = similarities_dir + 'user_user_' + similarity + '.csv'

        from recommender_systems.memory_based.user_based_CF import UserBasedCF as RS
        rs = RS(train, similarity_filename, similarity=similarity_function, notification_level=0, log=log if args.log else None)
        rs.getRecommendationsForAllUsers(n_neighbors=UKNN, verbose=True, output_filename=recommendation_filename, sep=',', top_n=TOP_N)

    elif rs_model == 'mfcf':
        from recommender_systems.model_based.matrix_factorization_CF import MatrixFactorizationCF as RS
        train_data, train_users, train_items = train
        mfcf_train_data, mfcf_train_user, mfcf_train_item = convert_to_matrix(train_data, train_users, train_items)

        rs = RS(mfcf_train_data, mfcf_train_user, mfcf_train_item, K=K, alpha=ALPHA, beta=BETA, iterations=MAX_ITER, notification_level=0, log=log if args.log else None)

        rs.train(verbose=True)
        rs.save_recommendations(output_path=recommendation_filename, n=TOP_N, verbose=True)

    else:
        if args.log:
            log.append('recommender system {} not found'.format(rs_model))
            log.abort()
        noti.balloon_tip('SAShA Detection', 'Recommender system {} not found. Experiment aborted.'.format(rs_model))
        raise ValueError('Recommender system not found.')
    
    # logging is done outside of this function
    pass

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
    
    try:
        next_version = np.array([int(name[len('experiment_results_'):]) for name in os.listdir(OUTDIR) if os.path.isdir(os.path.join(OUTDIR, name)) and name.startswith('experiment_results_')]).max() + 1
    except ValueError:
        next_version = 1

    dirname = OUTDIR + 'experiment_results_' + str(next_version) + '/'
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
        elif dataset == 'dummy':
            train, _ = load_data_dummy()
        else:
            if args.log:
                log.append('dataset {} not found'.format(dataset))
                log.abort()

            noti.balloon_tip('SAShA Detection', 'Dataset {} not found. Experiment aborted.'.format(dataset))
            raise ValueError('Dataset not found.')
        
        if args.log:
            log.append('dataset {} loaded'.format(dataset))

        # create directory for current dataset
        currentdir = dirname + dataset + '/'
        os.makedirs(currentdir, exist_ok=True)
        
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
        popular_items.to_csv(currentdir + '{}_popular_items.csv'.format(NUM_TARGET_ITEMS), index=False)
        print('generated {} popular items for dataset {}'.format(NUM_TARGET_ITEMS, dataset))
        if args.log:
            log.append('generated {} popular items for dataset {}. Saved in file {}'.format(NUM_TARGET_ITEMS, dataset, currentdir + '{}_popular_items.csv'.format(NUM_TARGET_ITEMS)))

        # list most unpopular items; to be used lates as target items of push attacks
        unpopular_items = items_sorted.tail(NUM_TARGET_ITEMS).iloc[::-1]
        unpopular_items.to_csv(currentdir + '{}_unpopular_items.csv'.format(NUM_TARGET_ITEMS), index=False)
        print('generated {} unpopular items for dataset {}'.format(NUM_TARGET_ITEMS, dataset))
        if args.log:
            log.append('generated {} unpopular items for dataset {}. Saved in file {}'.format(NUM_TARGET_ITEMS, dataset, currentdir + '{}_unpopular_items.csv'.format(NUM_TARGET_ITEMS)))

        # choose similarity measure --------------------------------------------------------------------------------------------------------
        pre_attack_similarities_dir = currentdir + 'similarities/pre_attack/'
        os.makedirs(pre_attack_similarities_dir, exist_ok=True)

        post_attack_similarities_dir = currentdir + 'similarities/post_attack/'
        os.makedirs(post_attack_similarities_dir, exist_ok=True)

        post_detection_similarities_dir = currentdir + 'similarities/post_detection/'
        os.makedirs(post_detection_similarities_dir, exist_ok=True)

        for similarity in SIMILARITY_MEASURES:
            print('Proceeding with similarity measure {}'.format(similarity))

            if args.log:
                log.append('Proceeding with similarity measure {}'.format(similarity))

            # choose recommender system -----------------------------------------------------------------------------------------------------
            for rs_model in RS_MODELS:

                print('Proceeding with recommender system {}'.format(rs_model))
                if args.log:
                    log.append('Proceeding with recommender system {}'.format(rs_model))

                # generate pre-attack recommendations ---------------------------------------------------------------------------------------
                recommendations_dir = currentdir + rs_model + '/recommendations/'
                os.makedirs(recommendations_dir, exist_ok=True)

                print('Generating pre-attack recommendations')
                if args.log:
                    log.append('Pre-attack recommendations generation initiated')

                pre_attack_recommendations_filename = recommendations_dir + 'pre_attack_{}_recommendations.csv'.format(similarity)
                generateRecommendations(train=train, 
                                        rs_model=rs_model, 
                                        similarity=similarity, 
                                        similarities_dir=pre_attack_similarities_dir, recommendation_filename=pre_attack_recommendations_filename, 
                                        log=log)
                
                print('Pre-attack recommendations for {} generated'.format(rs_model))
                if args.log:
                    log.append('Pre-attack recommendations for {} generated'.format(rs_model))

                noti.balloon_tip('SAShA Detection', 'Pre-attack recommendations for {} generated'.format(rs_model))

                # calculate pre-attack metrics ----------------------------------------------------------------------------------------------->>> LEFT OFF HERE
                # post_attack_recommendations_dir = recommendations_dir + attack + '/'

        noti.balloon_tip('SAShA Detection', 'Experiment dataset {} finished. Results are saved in {}'.format(dataset, currentdir))
    
    noti.balloon_tip('SAShA Detection', 'Experiment finished. Results are saved in {}'.format(dirname))


                


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', type=bool, default=True, help='verbose mode')
    parser.add_argument('--noti_level', type=int, default=0, help='notification level, 0: no notification, 1: only at the end, 2: at verbose mode')
    parser.add_argument('--log', type=bool, default=True, help='log mode')

    args = parser.parse_args()
    main()


