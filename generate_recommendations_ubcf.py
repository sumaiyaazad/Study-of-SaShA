import numpy as np
from utils.similarity_measures import *
from recommender_systems.memory_based.user_based_CF import UserBasedCF
from utils.data_loader import *
from config import *
import os

import argparse

def main():

    if args.dataset == 'ml-1m':
        data = load_data_ml_1M_ratings()
        user_data = load_data_ml_1M_users()
        item_data = load_data_ml_1M_items()
    # elif args.dataset == 'ml-100k':
    #     data = load_data_ml_100k_ratings()
    # elif args.dataset == 'ml-10m':
    #     data = load_data_ml_10M_ratings()
    # elif args.dataset == 'ml-20m':
    #     data = load_data_ml_20M_ratings()
    else:
        raise ValueError('Dataset not found.')
    
    
    train_data, test_data = train_test_split(data, test_size=0.02, train_size=0.08)



    # drop index
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    user_data = user_data.reset_index(drop=True)
    item_data = item_data.reset_index(drop=True)

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


    ubcf = UserBasedCF(train_data, test_data, user_data, item_data, n_users=args.n_users, n_items=args.n_items, similarity=similarity_measure)

    # output = OUTDIR + 'user_based_CF/' + 'user_based_CF_' + 'ml-1m' + 'recommendations.csv'
    output = OUTDIR + 'user_based_CF/' + 'user_based_CF_' + args.dataset + args.output_filename

    
    # create directory if it doesn't exist
    if not os.path.exists(OUTDIR + 'user_based_CF/'):
        os.makedirs(OUTDIR + 'user_based_CF/')


    # save similarity matrix if it doesn't exist
    if args.save_simi:
        ubcf.update_save_similarities(OUTDIR + 'user_based_CF/' + 'user_based_CF_' + args.save_simi_location)

    # load similarity matrix if it exists
    if args.load_simi:
        ubcf.loadSimilarities(OUTDIR + 'user_based_CF/' + 'user_based_CF_' + args.load_simi_location)
        # print(ubcf.user_user_similarity)


    ubcf.getRecommendationsForAllUsers(verbose=True, output_filename=output, sep=',', n_neighbors=args.n_neighbors)

    print()
    print('Experiment completed.')


if __name__ == '__main__':
    
    # main command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-1m', help='dataset to use')
    parser.add_argument('--n_users', type=int, default=None, help='number of users to use')
    parser.add_argument('--n_items', type=int, default=None, help='number of items to use')
    parser.add_argument('--n_neighbors', type=int, default=10, help='number of neighbors to use')
    parser.add_argument('--similarity_measure', type=str, default='cosine', help='similarity measure to use')
    parser.add_argument('--verbose', type=bool, default=False, help='verbose mode')
    parser.add_argument('--output_filename', type=str, default='recommendations.csv', help='output filename')
    parser.add_argument('--sep', type=str, default=',', help='separator for output file')
    parser.add_argument('--save_simi', type=bool, default=False, help='save similarity matrix')
    parser.add_argument('--load_simi', type=bool, default=False, help='load similarity matrix')
    parser.add_argument('--save_simi_location', type=str, default='user_user_similarity.pickle', help='location to save similarity matrix')
    parser.add_argument('--load_simi_location', type=str, default='user_user_similarity.pickle', help='location to load similarity matrix from')
    args = parser.parse_args()


    

    # print args
    # print('Command line arguments:')
    # print(args)

    print('*'*10, 'Starting experiment', '*'*10)
    print('Dataset: {}'.format(args.dataset))
    print('Number of users: {}'.format(args.n_users))
    print('Number of items: {}'.format(args.n_items))
    print('Number of neighbors: {}'.format(args.n_neighbors))
    print('Similarity measure: {}'.format(args.similarity_measure))
    print('Output directory: {}'.format(OUTDIR + 'user_based_CF/' + 'user_based_CF_' + args.dataset + args.output_filename))
    if args.save_simi:
        print('Saving similarity matrix to: {}'.format(args.save_simi_location))
    if args.load_simi:
        print('Loading similarity matrix from: {}'.format(args.load_simi_location))
    print()


    main()


