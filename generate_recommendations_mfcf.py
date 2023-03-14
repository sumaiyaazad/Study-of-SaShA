import numpy as np
from utils.similarity_measures import *
from recommender_systems.model_based.matrix_factorization_CF import MatrixFactorizationCF
from utils.data_loader import *
from config import *
import os
import matplotlib.pyplot as plt

import argparse

def main():

    if args.dataset == 'ml-1m':
        data = load_data_ml_1M_ratings()
    # elif args.dataset == 'ml-100k':
    #     data = load_data_ml_100k_ratings()
    # elif args.dataset == 'ml-10m':
    #     data = load_data_ml_10M_ratings()
    # elif args.dataset == 'ml-20m':
    #     data = load_data_ml_20M_ratings()
    else:
        raise ValueError('Dataset not found.')
    
    
    data = convert_to_matrix(data)

    if args.n_users is None:
        args.n_users = data.shape[0]

    if args.n_items is None:
        args.n_items = data.shape[1]

    data = data[:args.n_users, :args.n_items]

    print('number of users: ', args.n_users)
    print('number of items: ', args.n_items)
    print()


    
    path = OUTDIR + 'matrix_factorization_CF/' + 'matrix_factorization_CF_' + args.dataset + 'full_mat.csv'

    # create directory if it doesn't exist
    if not os.path.exists(OUTDIR + 'matrix_factorization_CF/'):
        os.makedirs(OUTDIR + 'matrix_factorization_CF/')

    # mf = MatrixFactorizationCF(data, K=2, alpha=0.001, beta=0.02, iterations=100)
    mf = MatrixFactorizationCF(data, K=args.k, alpha=args.alpha, beta=args.beta, iterations=args.max_iter)
    # mf = MatrixFactorizationCF(data, K=K, alpha=ALPHA, beta=BETA, iterations=MAX_ITER)

    mf.save_data_matrix(OUTDIR + 'matrix_factorization_CF/' + 'matrix_factorization_CF_' + args.dataset + 'data_matrix.csv')
    mses = mf.train(verbose=args.verbose)
    mf.save_full_matrix(path)

    # plot mse vs iterations

    x = [i for i,_ in mses]
    y = [j for _,j in mses]

    plt.plot(x, y)

    plt.xlabel("Iterations")
    plt.ylabel("MSE")
    plt.title("MSE vs Iterations")
    plt.savefig(OUTDIR + 'matrix_factorization_CF/' + 'matrix_factorization_CF_' + args.dataset + 'mse.png')

    print()
    print('Experiment completed.')


if __name__ == '__main__':
    
    # main command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-1m', help='dataset to use')
    parser.add_argument('--n_users', type=int, default=None, help='number of users to use')
    parser.add_argument('--n_items', type=int, default=None, help='number of items to use')
    parser.add_argument('--verbose', type=bool, default=False, help='verbose mode')
    parser.add_argument('--output_filename', type=str, default='full_matrix.csv', help='output filename')
    parser.add_argument('--k', type=int, default=2, help='number of latent factors')
    parser.add_argument('--alpha', type=float, default=0.001, help='learning rate')
    parser.add_argument('--beta', type=float, default=0.02, help='regularization parameter')
    parser.add_argument('--max_iter', type=int, default=50, help='maximum number of iterations')
    args = parser.parse_args()


    

    # print args
    # print('Command line arguments:')
    # print(args)

    print('*'*10, 'Starting experiment', '*'*10)
    print('Dataset: {}'.format(args.dataset))
    print('Number of users: {}'.format(args.n_users))
    print('Number of items: {}'.format(args.n_items))
    print('Number of latent factors: {}'.format(args.k))
    print('Learning rate: {}'.format(args.alpha))
    print('Regularization parameter: {}'.format(args.beta))
    print('Maximum number of iterations: {}'.format(args.max_iter))
    print()


    main()


