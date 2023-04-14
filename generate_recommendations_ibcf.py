import numpy as np
from utils.similarity_measures import *
from recommender_systems.memory_based.item_based_CF import ItemBasedCF
from utils.data_loader import *
from config import *
import os
from utils.notification import *

import argparse

def main():
    
    if args.dataset == 'ml-1m':
        # (data, user_data, item_data), _ = load_data_ml_1M(split=True)
        data, user_data, item_data = load_data_ml_1M()
    else:
        raise ValueError('Dataset not found.')
    
    
    # train_data, test_data = train_test_split(data, test_size=0.02, train_size=0.08)

    similarity_measure = adjusted_cosine_similarity

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

    ibcf = ItemBasedCF(data, user_data, item_data, n_users=args.n_users, n_items=args.n_items, similarity=similarity_measure, notification_level=args.not_level)

    # output = OUTDIR + 'user_based_CF/' + 'user_based_CF_' + 'ml-1m' + '_recommendations.csv'
    output = OUTDIR + 'item_based_CF/' + 'item_based_CF_' + args.dataset + '_' + args.output_filename

    # create directory if it doesn't exist
    if not os.path.exists(OUTDIR + 'item_based_CF/'):
        os.makedirs(OUTDIR + 'item_based_CF/')

    # save similarity matrix if it doesn't exist
    if args.save_simi:
        ibcf.update_save_similarities(OUTDIR + 'item_based_CF/' + 'item_based_CF_' + args.save_simi_location)

    # load similarity matrix if it exists
    if args.load_simi:
        ibcf.loadSimilarities(OUTDIR + 'item_based_CF/' + 'item_based_CF_' + args.load_simi_location)
        # print(ubcf.user_user_similarity)


    ibcf.getRecommendationsForAllUsers(verbose=True, output_filename=output, sep=',', n_neighbors=args.n_neighbors, top_n=args.top_n)

    print()
    print('Experiment completed.')
    balloon_tip('SAShA Detection', 'IBCF recommendation experiment completed.')


if __name__ == '__main__':
    
    # main command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-1m', help='dataset to use')
    parser.add_argument('--n_users', type=int, default=None, help='number of users to use')
    parser.add_argument('--n_items', type=int, default=None, help='number of items to use')
    parser.add_argument('--n_neighbors', type=int, default=10, help='number of neighbors to use')
    parser.add_argument('--top_n', type=int, default=50, help='top n recommendations to return')
    parser.add_argument('--similarity_measure', type=str, default='adjusted_cosine', help='similarity measure to use')
    parser.add_argument('--verbose', type=bool, default=False, help='verbose mode')
    parser.add_argument('--not_level', type=int, default=0, help='notification level, 0: no notification, 1: only at the end, 2: at verbose mode')
    parser.add_argument('--output_filename', type=str, default='recommendations.csv', help='output filename')
    parser.add_argument('--sep', type=str, default=',', help='separator for output file')
    parser.add_argument('--save_simi', type=bool, default=False, help='save similarity matrix')
    parser.add_argument('--load_simi', type=bool, default=False, help='load similarity matrix')
    parser.add_argument('--save_simi_location', type=str, default='item_item_similarity.pickle', help='location to save similarity matrix')
    parser.add_argument('--load_simi_location', type=str, default='item_item_similarity.pickle', help='location to load similarity matrix from')
    args = parser.parse_args()


    

    # print args
    # print('Command line arguments:')
    # print(args)

    print('*'*10, 'Starting experiment', '*'*10)
    print('Dataset: {}'.format(args.dataset))
    print('Number of users: {}'.format(args.n_users))
    print('Number of items: {}'.format(args.n_items))
    print('Number of neighbors: {}'.format(args.n_neighbors))
    print('Top n recommendations: {}'.format(args.top_n))
    print('Similarity measure: {}'.format(args.similarity_measure))
    print('Output directory: {}'.format(OUTDIR + 'item_based_CF/' + 'item_based_CF_' + args.dataset + args.output_filename))
    if args.save_simi:
        print('Saving similarity matrix to: {}'.format(args.save_simi_location))
    if args.load_simi:
        print('Loading similarity matrix from: {}'.format(args.load_simi_location))
    print()


    main()

