import numpy as np
from utils.similarity_measures import *
from recommender_systems.memory_based.user_based_CF import UserBasedCF
from utils.data_loader import *
from config import *
import os
from utils.notification import *

import argparse

def main():

    if args.dataset == 'ml-1m':
        train = load_data_ml_1M()
    elif args.dataset == 'dummy':
        train, _ = load_data_dummy()
    else:
        raise ValueError('Dataset not found.')
    
    
    # train_data, test_data = train_test_split(data, test_size=0.02, train_size=0.08)

    similarity_measure = cosine_similarity

    if args.similarity_measure == 'cosine':
        similarity_measure = cosine_similarity
    elif args.similarity_measure == 'jaccard':
        similarity_measure = jaccard_similarity
    elif args.similarity_measure == 'pearson':
        # can't yet be used
        similarity_measure = pearson_correlation
    elif args.similarity_measure == 'adjusted_cosine':
        similarity_measure = adjusted_cosine_similarity
    else:
        raise ValueError('Similarity measure not found.')



    # output = OUTDIR + 'user_based_CF/' + 'user_based_CF_' + 'ml-1m' + '_recommendations.csv'
    output = OUTDIR + 'user_based_CF/' + 'user_based_CF_' + args.dataset + '_' + args.output_filename

    
    # create directory if it doesn't exist
    if not os.path.exists(OUTDIR + 'user_based_CF/'):
        os.makedirs(OUTDIR + 'user_based_CF/')

    similarity_filename = OUTDIR + 'user_based_CF/' + 'user_based_CF_' + args.simi_location

    ubcf = UserBasedCF(train, similarity_filename, similarity=similarity_measure, rating_range=(1, 5), notification_level=args.not_level, log=None)

    ubcf.getRecommendationsForAllUsers(verbose=True, output_filename=output, sep=',', n_neighbors=args.n_neighbors, top_n=args.top_n)

    print()
    print('Experiment completed.')
    balloon_tip('SAShA Detection', 'UBCF recommendation experiment completed.')


if __name__ == '__main__':
    
    # main command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-1m', help='dataset to use')
    parser.add_argument('--n_users', type=int, default=None, help='number of users to use')
    parser.add_argument('--n_items', type=int, default=None, help='number of items to use')
    parser.add_argument('--n_neighbors', type=int, default=10, help='number of neighbors to use')
    parser.add_argument('--top_n', type=int, default=50, help='number of top recommendations to generate')
    parser.add_argument('--similarity_measure', type=str, default='cosine', help='similarity measure to use')
    parser.add_argument('--verbose', type=bool, default=False, help='verbose mode')
    parser.add_argument('--not_level', type=int, default=0, help='notification level, 0: no notification, 1: only at the end, 2: at verbose mode')
    parser.add_argument('--output_filename', type=str, default='recommendations.csv', help='output filename')
    parser.add_argument('--sep', type=str, default=',', help='separator for output file')
    parser.add_argument('--simi_location', type=str, default='user_user_similarity.csv', help='location to save similarity matrix')
    args = parser.parse_args()


    print('*'*10, 'Starting experiment', '*'*10)
    print('Dataset: {}'.format(args.dataset))
    print('Number of users: {}'.format(args.n_users))
    print('Number of items: {}'.format(args.n_items))
    print('Number of neighbors: {}'.format(args.n_neighbors))
    print('Number of top recommendations: {}'.format(args.top_n))
    print('Similarity measure: {}'.format(args.similarity_measure))
    print('Output directory: {}'.format(OUTDIR + 'user_based_CF/' + 'user_based_CF_' + args.dataset + args.output_filename))
    
    print()


    main()


