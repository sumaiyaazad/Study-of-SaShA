# Attempting to decouple the steps of the experiment to make it fail safe
# things added:
# - try/except blocks
# - logging breakpoints when failures occur
# - restarts from the last successful step

import argparse
import pandas as pd
import os

from config import *

import utils.notification as noti
from utils.data_loader import *
from utils.similarity_measures import *
from utils.misc import *
from utils.log import Logger
from utils.evaluation import *
from utils.sendmail import sendmail, sendmailwithfile

global BREAKPOINT
BREAKPOINT = 0


def load_data(dataset, dirname, all_data, all_currentdir, log):
    """
    Load dataset
    param dataset: name of the dataset
    param dirname: directory name
    param all_data: dictionary of all datasets
    param all_currentdir: dictionary of all current directories
    param log: object of Logger class
    """
    
    # ------------------------------------------------------------------------------------------------ load dataset
    if dataset in all_data.keys():
        data = all_data[dataset]
    elif dataset == 'ml-1m':
        data = load_data_ml_1M(split=True)
    elif dataset == 'dummy':
        data = load_data_dummy()
    else:
        if args.log:
            log.append('dataset {} not found'.format(dataset))
            log.abort()
        raise ValueError('Dataset not found.')
    if args.log:
        log.append('dataset {} loaded'.format(dataset))
    all_data[dataset] = data

    # create directory for current dataset
    if dataset in all_currentdir.keys():
        currentdir = all_currentdir[dataset]
    else:
        currentdir = dirname + dataset + '/'
        all_currentdir[dataset] = currentdir
    os.makedirs(currentdir, exist_ok=True)

    return all_data, all_currentdir, data, currentdir

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
        if args.noti_level > 0:
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
        if args.noti_level > 0:
            noti.balloon_tip('SAShA Detection', 'Recommender system {} not found. Experiment aborted.'.format(rs_model))
        raise ValueError('Recommender system not found.')
    
    # logging is done outside of this function
    pass


def experiment(log, dirname):
    """
    Run experiment
    param log: object of Logger class
    param dirname: directory to store experiment results
    """
    if args.log:
        log.append('experiment started')
        log.append('dataset: {}'.format(DATASETS))
        log.append('recommender system: {}'.format(RS_MODELS))
        log.append('similarity measure: {}'.format(SIMILARITY_MEASURES))
        log.append('attack: {}'.format(ATTACKS))
        log.append('attack impact evaluation metrics: {}'.format(EVALUATIONS))
        log.append('detection: {}'.format(DETECTIONS))
        log.append('experiment start time: {}'.format(now()))
        log.append('experiment result directory: {}'.format(dirname))
        log.append('Log file: {}'.format(LOG_FILE))
        log.append('\n\n\n')

    if args.send_mail:
        sendmail('SAShA Detection', 'Experiment started')



    BREAKPOINT = 1  # ------------------------------------------------------------------------------------ breakpoint 1 (experiment start)
    print('BREAKPOINT 1')
    bigskip()
    if args.log:
        log.append('BREAKPOINT 1')
        log.append('\n\n\n')

    all_data = {}
    all_currentdir = {}

    for dataset in DATASETS:
        # load dataset
        all_data, all_currentdir, data, currentdir = load_data(dataset, dirname, all_data, all_currentdir, log)
        
        train, test = data
        train_data, train_users, train_items = train

        # sort items by average rating
        items_sorted = train_data.groupby('item_id')['rating'].mean().to_frame()
        items_sorted.reset_index(inplace=True)
        items_sorted = items_sorted.rename(columns = {'index':'item_id', 'rating':'avg_rating'})
        items_sorted = items_sorted.sort_values(by=['avg_rating'], ascending=False)
        items_sorted.reset_index(inplace=True)
        items_sorted = items_sorted.drop(columns=['index'])

        # -------------------------------------------------------------------------------------------------------  list most popular items
        popular_items = items_sorted.head(NUM_TARGET_ITEMS)
        popular_items.to_csv(currentdir + '{}_popular_items.csv'.format(NUM_TARGET_ITEMS), index=False)
        print('generated {} popular items for dataset {}'.format(NUM_TARGET_ITEMS, dataset))
        if args.log:
            log.append('generated {} popular items for dataset {}. Saved in file {}'.format(NUM_TARGET_ITEMS, dataset, currentdir + '{}_popular_items.csv'.format(NUM_TARGET_ITEMS)))

        # --------------------------------------------------------------------------------------------------------- list most unpopular items; to be used lates as target items of push attacks
        unpopular_items = items_sorted.tail(NUM_TARGET_ITEMS).iloc[::-1]
        unpopular_items.to_csv(currentdir + '{}_unpopular_items.csv'.format(NUM_TARGET_ITEMS), index=False)
        print('generated {} unpopular items for dataset {}'.format(NUM_TARGET_ITEMS, dataset))
        if args.log:
            log.append('generated {} unpopular items for dataset {}. Saved in file {}'.format(NUM_TARGET_ITEMS, dataset, currentdir + '{}_unpopular_items.csv'.format(NUM_TARGET_ITEMS)))


# so far, we have generated the most popular and unpopular items for the dataset
# now, we will generate pre-attack recommendations for each recommender system


    BREAKPOINT = 2  # ------------------------------------------------------------------------------------ breakpoint 2 (generate pre-attack recommendations)
    print('BREAKPOINT 2')
    bigskip()
    if args.log:
        log.append('BREAKPOINT 2')
        log.append('\n\n\n')

    for dataset in DATASETS:
        # load dataset
        all_data, all_currentdir, data, currentdir = load_data(dataset, dirname, all_data, all_currentdir, log)
        
        train, test = data
        train_data, train_users, train_items = train

        # create directory for storing similarities
        pre_attack_similarities_dir = currentdir + 'similarities/pre_attack/'
        os.makedirs(pre_attack_similarities_dir, exist_ok=True)
        if args.log:
            log.append('created directory {}'.format(pre_attack_similarities_dir))

        post_attack_similarities_dir = currentdir + 'similarities/post_attack/'
        os.makedirs(post_attack_similarities_dir, exist_ok=True)
        if args.log:
            log.append('created directory {}'.format(post_attack_similarities_dir))

        post_detection_similarities_dir = currentdir + 'similarities/post_detection/'
        os.makedirs(post_detection_similarities_dir, exist_ok=True)
        if args.log:
            log.append('created directory {}'.format(post_detection_similarities_dir))

        for similarity in SIMILARITY_MEASURES:
            for rs_model in RS_MODELS:

                # generate pre-attack recommendations ---------------------------------------------------------------------------------------
                recommendations_dir = currentdir + rs_model + '/recommendations/'
                os.makedirs(recommendations_dir, exist_ok=True)

                print('Generating pre-attack recommendations for {} with {} similarity for dataset {}'.format(rs_model, similarity, dataset))
                if args.log:
                    log.append('Generating pre-attack recommendations for {} with {} similarity for dataset {}'.format(rs_model, similarity, dataset))


                pre_attack_recommendations_filename = recommendations_dir + 'pre_attack_{}_recommendations.csv'.format(similarity)
                generateRecommendations(train=train, 
                                        rs_model=rs_model, 
                                        similarity=similarity, 
                                        similarities_dir=pre_attack_similarities_dir, recommendation_filename=pre_attack_recommendations_filename, 
                                        log=log)
                
                print('Pre-attack recommendations for {} with {} similarity for dataset {} generated'.format(rs_model, similarity, dataset))
                if args.log:
                    log.append('Pre-attack recommendations for {} with {} similarity for dataset {} generated'.format(rs_model, similarity, dataset))

                if args.send_mail:
                    sendmail(SUBJECT, 'Pre-attack recommendations for generated.\nDataset: {}\nRS: {}\nSimilarity: {}\n'.format(dataset, rs_model, similarity))

                if args.noti_level > 0:
                    noti.balloon_tip('SAShA Detection', 'Pre-attack recommendations for {} generated'.format(rs_model))

            
# so far we have generated pre-attack recommendations for each recommender system
# now, we will calculate the hit ratio of the pre-attack recommendations

    BREAKPOINT = 3  # ------------------------------------------------------------------------------------ breakpoint 3 (calculate hit ratio of pre-attack recommendations)
    print('BREAKPOINT 3')
    bigskip()
    if args.log:
        log.append('BREAKPOINT 3')
        log.append('\n\n\n')

    for dataset in DATASETS:
        
        if dataset in all_currentdir.keys():
            currentdir = all_currentdir[dataset]
        else:
            currentdir = dirname + dataset + '/'
            all_currentdir[dataset] = currentdir

        for similarity in SIMILARITY_MEASURES:
            for rs_model in RS_MODELS:

                # calculate hit ratio of pre-attack recommendations ---------------------------------------------------------------------------------------
                recommendations_dir = currentdir + rs_model + '/recommendations/'
                hit_ratio_dir = currentdir + rs_model + '/results/' + 'hit_ratio/'
                os.makedirs(hit_ratio_dir, exist_ok=True)

                print('Calculating hit ratio of pre-attack recommendations for {} with {} similarity for dataset {}'.format(rs_model, similarity, dataset))
                if args.log:
                    log.append('Calculating hit ratio of pre-attack recommendations for {} with {} similarity for dataset {}'.format(rs_model, similarity, dataset))

                # load target items
                target_items = pd.read_csv(currentdir + '{}_unpopular_items.csv'.format(NUM_TARGET_ITEMS))
                target_items.columns = ['item_id', 'avg_rating']
                target_items = target_items['item_id'].tolist()

                pre_attack_recommendations_filename = recommendations_dir + 'pre_attack_{}_recommendations.csv'.format(similarity)
                pre_attack_hit_ratio = hit_ratio(recommendations_filename = pre_attack_recommendations_filename,
                                                 target_items = target_items,
                                                 amoung_firsts=TOP_Ns,
                                                 log = log)
                
                pre_attack_hit_ratio.to_csv(hit_ratio_dir + 'pre_attack_{}_hit_ratio.csv'.format(similarity), index=False)
                print('Hit ratio of pre-attack recommendations for {} with {} similarity for dataset {} calculated'.format(rs_model, similarity, dataset))
                if args.log:
                    log.append('Hit ratio of pre-attack recommendations for {} with {} similarity for dataset {} calculated'.format(rs_model, similarity, dataset))

    if args.send_mail:
        sendmail(SUBJECT, 'Hit ratio of pre-attack recommendations calculated.')

# so far we have calculated the hit ratio of pre-attack recommendations
# now, we will launch attacks

    BREAKPOINT = 4  # ------------------------------------------------------------------------------------ breakpoint 4 (launch attacks)
    print('BREAKPOINT 4')
    bigskip()
    if args.log:
        log.append('BREAKPOINT 4')
        log.append('\n\n\n')


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



    # ------------------------------------------ starting experiment ------------------------------------------
    try:
        experiment(log, dirname)
    except Exception as e:
        if args.log:
            log.append('experiment failed')
            log.append('error: {}'.format(e))
            log.abort()
        
        if args.send_mail:
            email_body = 'Experiment failed.\r\nError: {}'.format(e)
            sendmailwithfile('SAShA Detection', email_body, 'log.txt', LOG_FILE)

        if args.noti_level > 0:
            noti.balloon_tip('SAShA Detection', 'Experiment failed. Error: {}'.format(e))
        raise e
    
    # ------------------------------------------ experiment finished ------------------------------------------

    print('experiment finished')
    
    if args.log:
        log.append('experiment finished')
    
    if args.send_mail:
        sendmailwithfile(subject=SUBJECT, message='Experiment finished Successfully. Results are saved in {}'.format(dirname), filelocation=LOG_FILE, filename='log.txt')
    
    if args.noti_level > 0:
        noti.balloon_tip('SAShA Detection', 'Experiment finished. Results are saved in {}'.format(dirname))
                


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', type=bool, default=True, help='verbose mode')
    parser.add_argument('--noti_level', type=int, default=0, help='notification level, 0: no notification, 1: only at the end, 2: at verbose mode')
    parser.add_argument('--log', type=bool, default=True, help='log mode')
    parser.add_argument('--send_mail', type=bool, default=True, help='send mail mode')

    args = parser.parse_args()

    main()


