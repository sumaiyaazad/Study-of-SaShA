import random
from utils.data_loader import *
import time
from tqdm import tqdm
import numpy as np
import math as m
import pandas as pd
from utils.similarity_measures import *
import pickle
from utils.notification import *


# implementation of item based collaborative filtering

class UserBasedCF:

    def __init__(self, train_data, user_data, item_data, n_users=None, n_items=None, similarity=cosine_similarity, notification_level=0):

        if n_users is None:
            self.train_data = train_data
            # self.test_data = test_data

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
            # self.test_data = test_data.loc[test_data['user_id'].isin(self.user_data['user_id']) & test_data['item_id'].isin(self.item_data['item_id'])]
            
        self.similarity = similarity
        self.userItemMatrix = None
        self.user_user_similarity = None
        self.recommendations = None

        self.save_similarities = None
        self.notification_level = notification_level

    def update_save_similarities(self, filename):
        self.save_similarities = filename

    def saveSimilarities(self):
        if self.save_similarities is None:
            print('No filename given to save similarities')
            return

        print('*'*10, 'Saving similarities...', '*'*10)

        # pickle dump
        with open(self.save_similarities, 'wb') as handle:
            pickle.dump(self.user_user_similarity, handle)

        
        # save as csv
        # pd.DataFrame(self.user_user_similarity).to_csv(self.save_similarities, index=False)
        
        print('Similarities saved to {}'.format(self.save_similarities))

    def loadSimilarities(self, filename=None):
        if self.save_similarities is None:
            if filename is None:
                print('No filename given to save similarities')
                return
            else:
                self.save_similarities = filename

        print('*'*10, 'Loading similarities...', '*'*10)

        # load 
        with open(self.save_similarities, 'rb') as handle:
            self.user_user_similarity = pickle.load(handle)

        
        # load as csv
        # self.user_user_similarity = pd.read_csv(self.save_similarities).to_dict()

        print('Similarities loaded from {}'.format(self.save_similarities))            


    def getUserItemMatrix(self, verbose=False):
        # create user-item matrix dictionary
        self.userItemMatrix = {}

        if verbose:
            print('*'*10, 'Creating user-item matrix...', '*'*10)
            start_time = time.time()

        for user in self.user_data['user_id']:
            self.userItemMatrix.setdefault(user, {})

        for index, datapoint in tqdm(self.train_data.iterrows()):
            self.userItemMatrix.setdefault(datapoint['user_id'], {})
            self.userItemMatrix[datapoint['user_id']][datapoint['item_id']] = datapoint['rating']

        if verbose:
            print('Time taken: {:.2f} seconds'.format(time.time() - start_time))
            print('User-item matrix created.')
            print()
            if self.notification_level >= 2:
                balloon_tip('SAShA Detection', 'User-item matrix created.')

    def getUserPairSimilarity(self, user1, user2):
        '''
        Calculate similarity between two users
        this function is not used in the code. slow
        '''


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

        return self.similarity(user1_common_ratings, user2_common_ratings)
    
    def getUserUserSimilarity(self, verbose=False):

        if self.userItemMatrix is None:
            self.getUserItemMatrix(verbose=verbose)

        # create user-user similarity matrix dictionary
        self.user_user_similarity = {}

        if verbose:
            print('*'*10, 'Creating user-user similarity matrix...', '*'*10)
            start_time = time.time()

        for user1 in tqdm(self.user_data['user_id'].unique()):
            self.user_user_similarity.setdefault(user1, {})
            for user2 in self.user_data['user_id'].unique():
                if user1 != user2:
                    self.user_user_similarity[user1][user2] = self.similarity(self.userItemMatrix[user1], self.userItemMatrix[user2])
                    # self.user_user_similarity[user1][user2] = self.getUserPairSimilarity(user1, user2)
                
        if verbose:
            print('Time taken: {:.2f} seconds'.format(time.time() - start_time))
            print('User-user similarity matrix created.')
            print()
            if self.notification_level >= 2:
                balloon_tip('SAShA Detection', 'User-user similarity matrix created.')


        if self.save_similarities is not None:
            print('Saving similarities...')
            self.saveSimilarities()
            print('Similarities saved.')

    def getRecommendations(self, user_id, n_neighbors=10, verbose=False):

        if self.user_user_similarity is None:
            self.getUserUserSimilarity(verbose=verbose)

        if self.userItemMatrix is None:
            self.getUserItemMatrix(verbose=verbose)

        if verbose:
            print('*'*10, 'Getting recommendations for user {}...'.format(user_id), '*'*10)
            start_time = time.time()

        # get top k similar users
        try:
            top_k_similar_users = sorted(self.user_user_similarity[user_id].items(), key=lambda x: x[1], reverse=True)[:n_neighbors]
        except:
            print('debug')
            print(self.user_user_similarity[user_id].items())

        # top_k_similar_users = sorted(self.user_user_similarity[user_id].items(), key=lambda x: x[1], reverse=True)[:n_neighbors]

        # print('Top {} similar users: {}'.format(n_neighbors, top_k_similar_users))

        # get items rated by these users
        items_rated_by_similar_users = {}
        norm_factor = {}
        for similar_user in top_k_similar_users:
            for item in self.userItemMatrix[similar_user[0]]:
                items_rated_by_similar_users.setdefault(item, 0)
                norm_factor.setdefault(item, 0)

                # weighted by user user similarity
                items_rated_by_similar_users[item] += similar_user[1] * self.userItemMatrix[similar_user[0]][item]
                norm_factor[item] += similar_user[1]

        # normalize
        for item in items_rated_by_similar_users:
            if norm_factor[item] != 0:
                items_rated_by_similar_users[item] = items_rated_by_similar_users[item] / norm_factor[item]
        
        # sort items by similarity score
        items_rated_by_similar_users = sorted(items_rated_by_similar_users.items(), key=lambda x: x[1], reverse=True)

        # remove items already rated by user
        items_to_recommend = items_rated_by_similar_users.copy()
        for item in items_rated_by_similar_users:
            if item[0] in self.userItemMatrix[user_id]:
                items_to_recommend.remove(item)

        if verbose:
            print('Time taken: {:.2f} seconds'.format(time.time() - start_time))
            print('Recommendations for user {} are:'.format(user_id))
            print()
            if self.notification_level >= 1:
                balloon_tip('SAShA Detection', 'Recommendations for user {} are:'.format(user_id))

        return items_to_recommend
    
    def getRecommendationsForAllUsers(self, n_neighbors=10, verbose=False, output_filename='output/recommendations.csv', sep='::', top_n=None):

        if self.user_user_similarity is None:
            self.getUserUserSimilarity(verbose=verbose)

        if verbose:
            print('*'*10, 'Getting recommendations for all users...', '*'*10)
            start_time = time.time()
        
        self.recommendations = {}
        for user in tqdm(self.user_data['user_id']):
            self.recommendations[user] = self.getRecommendations(user, n_neighbors=n_neighbors)
        
        if verbose:
            print('Time taken: {:.2f} seconds'.format(time.time() - start_time))
            print('Recommendations for all users generated')
            print()
            if self.notification_level >= 1:
                balloon_tip('SAShA Detection', 'Recommendations for all users generated') 

        # write to file
        
        
        # write the recommendation items to file
        if verbose:
            print("Write recommendations to file...")

        with open(output_filename, "w") as f:
            for user, items in self.recommendations.items():
                if top_n is not None:
                    items = sorted(items, key=lambda x: x[1], reverse=True)
                    items = items[:top_n]
                for item, rating in items:
                    f.write(str(user) + sep + str(item) + sep + str(rating) + "\n")

        
        return self.recommendations
    