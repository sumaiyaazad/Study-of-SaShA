import numpy as np
from utils.similarity_measures import *
from recommender_systems.model_based.matrix_factorization_CF import MatrixFactorizationCF
from utils.data_loader import *
from config import *
import os
import matplotlib.pyplot as plt
from utils.notification import *
from utils.misc import convert_to_matrix

import argparse

def main():

    if args.dataset == 'ml-1m':
        data, user_data, item_data = load_data_ml_1M()
    elif args.dataset == 'dummy':
        (data, user_data, item_data), _ = load_data_dummy()
    else:
        raise ValueError('Dataset not found.')
    
    # converting data to matrix (user_index, item_index) = rating
    data, user_data, item_data = convert_to_matrix(data, user_data, item_data)

    # if args.n_users is None:
    #     args.n_users = data.shape[0]

    # if args.n_items is None:
    #     args.n_items = data.shape[1]

    # data = data[:args.n_users, :args.n_items]

    args.n_users = data.shape[0]
    args.n_items = data.shape[1]

    print('number of users: ', args.n_users)
    print('number of items: ', args.n_items)
    print()


    
    path = OUTDIR + 'matrix_factorization_CF/' + 'matrix_factorization_CF_' + args.dataset + '_full_mat.csv'

    # create directory if it doesn't exist
    if not os.path.exists(OUTDIR + 'matrix_factorization_CF/'):
        os.makedirs(OUTDIR + 'matrix_factorization_CF/')

    # mf = MatrixFactorizationCF(data, K=2, alpha=0.001, beta=0.02, iterations=100)
    mf = MatrixFactorizationCF(data, user_data, item_data, K=args.k, alpha=args.alpha, beta=args.beta, iterations=args.max_iter, rating_range=(1, 5), notification_level=args.not_level)
    # mf = MatrixFactorizationCF(data, K=K, alpha=ALPHA, beta=BETA, iterations=MAX_ITER)

    mf.save_data_matrix(OUTDIR + 'matrix_factorization_CF/' + 'matrix_factorization_CF_' + args.dataset + '_data_matrix.csv')
    mses = mf.train(verbose=args.verbose)
    mf.save_full_matrix(path)
    mf.save_recommendations(OUTDIR + 'matrix_factorization_CF/' + 'matrix_factorization_CF_' + args.dataset + '_recommendations.csv', n=10, verbose=args.verbose)

    # plot mse vs iterations

    x = [i for i,_ in mses]
    y = [j for _,j in mses]

    plt.plot(x, y)

    plt.xlabel("Iterations")
    plt.ylabel("MSE")
    plt.title("MSE vs Iterations")
    plt.savefig(OUTDIR + 'matrix_factorization_CF/' + 'matrix_factorization_CF_' + args.dataset + '_mse.png')

    print()
    print('Experiment completed.')
    balloon_tip('SAShA Detection', 'MFCF recommendation generation experiment completed.')


if __name__ == '__main__':
    
    # main command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-1m', help='dataset to use')
    parser.add_argument('--n_users', type=int, default=None, help='number of users to use')
    parser.add_argument('--n_items', type=int, default=None, help='number of items to use')
    parser.add_argument('--top_n', type=int, default=50, help='top n recommendations')
    parser.add_argument('--verbose', type=bool, default=True, help='verbose mode')
    parser.add_argument('--not_level', type=int, default=0, help='notification level, 0: no notification, 1: only at the end, 2: at verbose mode')
    parser.add_argument('--output_filename', type=str, default='full_matrix.csv', help='output filename')
    parser.add_argument('--k', type=int, default=5, help='number of latent factors')
    parser.add_argument('--alpha', type=float, default=0.001, help='learning rate')
    parser.add_argument('--beta', type=float, default=0.02, help='regularization parameter')
    parser.add_argument('--max_iter', type=int, default=100, help='maximum number of iterations')
    args = parser.parse_args()


    

    # print args
    # print('Command line arguments:')
    # print(args)

    print('*'*10, 'Starting experiment', '*'*10)
    print('Dataset: {}'.format(args.dataset))
    print('Number of users: {}'.format(args.n_users))
    print('Number of items: {}'.format(args.n_items))
    print('Top n recommendations: {}'.format(args.top_n))
    print('Number of latent factors: {}'.format(args.k))
    print('Learning rate: {}'.format(args.alpha))
    print('Regularization parameter: {}'.format(args.beta))
    print('Maximum number of iterations: {}'.format(args.max_iter))
    print()


    main()


