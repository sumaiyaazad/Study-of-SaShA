import numpy as np
import math as m
import pandas as pd
from operator import itemgetter
from utils.similarity_measures import *
import random
from utils.data_loader import *
import time
from tqdm import tqdm


# implementation of item based collaborative filtering


class ItemBasedCF:

    def __init__(self, train_data, test_data, user_data, item_data, n_users=None, n_items=None):

        if n_users is None:
            self.train_data = train_data
            self.test_data = test_data

            self.user_data = user_data
            self.item_data = item_data

            # pandas dataframe unique user and item
            self.n_users = len(self.user_data['user_id'].unique())
            self.n_items = len(self.item_data['item_id'].unique())

        else:
            self.n_users = n_users
            self.n_items = n_items
            
            self.user_data = user_data[:n_users]
            self.item_data = item_data[:n_items]
            
            self.train_data = train_data.loc[train_data['user_id'].isin(self.user_data['user_id']) & train_data['item_id'].isin(self.item_data['item_id'])]
            self.test_data = test_data.loc[test_data['user_id'].isin(self.user_data['user_id']) & test_data['item_id'].isin(self.item_data['item_id'])]
            
        
        self.userItemMatrix = None
        self.user_user_similarity = None

    
    def getUserItemMatrix(self, verbose=False):
        # create user-item matrix dictionary
        self.userItemMatrix = {}

        if verbose:
            print('*'*10, 'Creating user-item matrix...', '*'*10)
            start_time = time.time()

        for index, datapoint in self.train_data.iterrows():
            self.userItemMatrix.setdefault(datapoint['user_id'], {})
            self.userItemMatrix[datapoint['user_id']][datapoint['item_id']] = datapoint['rating']

        if verbose:
            print('Time taken: {:.2f} seconds'.format(time.time() - start_time))
            print('User-item matrix created.')
            print()

        return self.userItemMatrix

    

    def getUserPairSimilarity(self, user1, user2, simiarity=cosine_similarity):
        # calculate similarity between two users
        user1_ratings = self.train_data.loc[self.train_data['user_id'] == user1]
        user2_ratings = self.train_data.loc[self.train_data['user_id'] == user2]

        # common items rated by both users
        common_items_user1 = user1_ratings.loc[user1_ratings['item_id'].isin(user2_ratings['item_id'])]
        common_items_user2 = user2_ratings.loc[user2_ratings['item_id'].isin(user1_ratings['item_id'])]

        # sort by item_id
        common_items_user1 = common_items_user1.sort_values(by=['item_id'])
        common_items_user2 = common_items_user2.sort_values(by=['item_id'])

        # rating vectors of common items
        user1_common_ratings = common_items_user1['rating'].values
        user2_common_ratings = common_items_user2['rating'].values

        return simiarity(user1_common_ratings, user2_common_ratings)
        
    def getUserSimilarity(self, simiarity=cosine_similarity, verbose=False):

        if verbose:
            print('*'*10, 'Calculating user-user similarity matrix...', '*'*10)
            start_time = time.time()

        self.user_user_similarity = {}
        
        for user1 in tqdm(self.user_data['user_id'].unique()):
            self.user_user_similarity.setdefault(user1, {})
            for user2 in self.user_data['user_id'].unique():
                self.user_user_similarity[user1][user2] = self.getUserPairSimilarity(user1, user2, simiarity)
        
        if verbose:
            print('*'*10, 'Done!', '*'*10)
            print('Time taken: ', time.time() - start_time)
            print()

        return self.user_user_similarity

    def getRecommendations(self, user, k=10, verbose=False):
        if self.user_user_similarity is None:
            self.getUserSimilarity(verbose=verbose)

        if self.userItemMatrix is None:
            self.getUserItemMatrix(verbose=verbose)


        # find K nearest neighbors (users)
        nearest = sorted(self.user_user_similarity[int(user)].items(), key=lambda x:x[1], reverse=True)[:k]

        rank = dict()
        for neighbor in nearest:
            neighbor_user = neighbor[0]
            neighbor_similarity = neighbor[1]

            if neighbor_similarity == 0:
                continue
            if neighbor_user == user:
                continue
            if neighbor_user not in self.userItemMatrix:
                continue

            for item in self.userItemMatrix[neighbor_user]:
                if item not in self.userItemMatrix[user]:
                    rank.setdefault(item, 0)
                    rank[item] += neighbor_similarity * self.userItemMatrix[neighbor_user][item]

        return rank

    def evaluate(self, k=10):
        # return the recommendation items for each user
        self.getUserSimilarity()
        self.recommendations = dict()
        for user in self.train.keys():
            rank = self.recommend(user)
            self.recommendations[user] = dict(sorted(rank.items(), key=itemgetter(1), reverse=True)[0:k])

        # calculate precision and recall
        hit = 0
        precision = 0
        recall = 0
        for user in self.train.keys():
            # sort test items by rating in descending order
            test_recom = dict(sorted(self.test[user].items(), key=itemgetter(1), reverse=True)[:k])

            hit += set.intersection(set(self.recommendations[user].keys()), set(test_recom.keys())).__len__()
            precision += k
            recall += len(self.test[user])
        return (hit / (precision * 1.0)), (hit / (recall * 1.0))


    # generate the recommendation items for each user
    def getRecommendationsForAllUsers(self, num_of_recommendations=10, verbose=False, output_file='item_based_CF_recommendations.dat'):

        if self.user_user_similarity is None:
            self.getUserSimilarity(verbose=verbose)

        if self.userItemMatrix is None:
            self.getUserItemMatrix(verbose=verbose)

        if verbose:
            print('*'*10, 'Generating recommendations...', '*'*10)
            start_time = time.time()

        self.getUserSimilarity()
        self.recommendations = dict()
        
        for user in tqdm(self.user_data['user_id'].unique()):
            rank = self.recommend(user)
            self.recommendations[user] = dict(sorted(rank.items(), key=itemgetter(1), reverse=True)[0:num_of_recommendations])

        if verbose:
            print('*'*10, 'Done!', '*'*10)
            print('Time taken: ', time.time() - start_time)
            print()

        # write the recommendation items to file
        if verbose:
            print("Write recommendations to file...")

        with open(output_file, "w") as f:
            for user, items in self.recommendations.items():
                for item, rating in items.items():
                    f.write(str(user) + "::" + str(item) + "::" + str(rating) + "\n")