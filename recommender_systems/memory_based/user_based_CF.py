from utils.data_loader import *
from utils.notification import *
from utils.similarity_measures import *

import time
from tqdm import tqdm
import pandas as pd


# implementation of item based collaborative filtering

class UserBasedCF:

    def __init__(self, data, similarity_filename, similarity=cosine_similarity, notification_level=0, log=None):


        """
        Initialize the UserBasedCF class
        :param data: train data (train_data, train_users, train_items)
        :param similarity: similarity measure to use
        :param similarity_filename: filename to save/load item item similarities
        :param notification_level: 
                                    0: no notification, 
                                    1: notification when training is done and recommendations are saved,
                                    2: notification whatever verbosed :3
        :param log: object of class Logger
        """

        if log is not None:
            log.append('Creating object of UserBasedCF class')

        self.train_data, self.train_users, self.train_items = data

        self.n_users = len(self.train_users['user_id'].unique())
        self.n_items = len(self.train_items['item_id'].unique())

        self.similarity = similarity
        self.userItemMatrix = None
        self.user_user_similarity = None
        self.recommendations = None
        self.similarities_filename = similarity_filename
        self.notification_level = notification_level
        self.log = log

        self.loadSimilarities()


    def update_similarities_filename(self, filename):
        self.similarities_filename = filename

        if self.log is not None:
            self.log.append('Similarities filename updated to {}'.format(filename))

    def saveSimilarities(self):
        
        '''
        Save item item similarities to file
        (item1, item2, similarity)
        '''

        if self.log is not None:
            self.log.append('saving similarities to {} initiated'.format(self.similarities_filename))

        if self.similarities_filename is None:
            print('No filename given to save similarities')
            if self.log is not None:
                self.log.append('No filename given to save similarities')
                self.log.abort()
            raise ValueError('No filename given to save similarities')

        if self.user_user_similarity is None:
            self.getUserUserSimilarity(verbose=True)

        print('*'*10, 'Saving similarities...', '*'*10)

        # convert to list of tuples
        uusim = []

        for user1 in tqdm(self.train_users['user_id'].unique()):
            for user2 in self.train_users['user_id'].unique():
                if user1 != user2:
                    uusim.append([user1, user2, self.user_user_similarity[user1][user2]])

        # to dataframe
        uusim = pd.DataFrame(uusim, columns=['user1', 'user2', 'similarity'])

        # save as csv
        uusim.to_csv(self.similarities_filename, index=False)

        if self.log is not None:
            self.log.append('Similarities saved to {}'.format(self.similarities_filename))

        print('Similarities saved to {}'.format(self.similarities_filename))


    def loadSimilarities(self, verbose=False):

        if self.log is not None:
            self.log.append('Loading similarities initiated')

        if self.similarities_filename is None:
            print('No filename given to save similarities')
            if self.log is not None:
                self.log.append('No filename given to save similarities')
                self.log.abort()
            raise ValueError('No filename given to load similarities')

        print('*'*10, 'Loading similarities...', '*'*10)

        # load as csv
        try:
            uusim_df = pd.read_csv(self.similarities_filename, header=None, names=['user1', 'user2', 'similarity'])
        except FileNotFoundError:
            print('WARNING:File not found. Similarities will be calculated and saved to {}'.format(self.similarities_filename))
            if self.log is not None:
                self.log.append('WARNING:File not found. Similarities will be calculated and saved to {}'.format(self.similarities_filename))
            self.getUserUserSimilarity(verbose=True)
            return
        
        for user in self.train_users['user_id'].unique():
            self.user_user_similarity.setdefault(user, {})
            
        for user1, user2, sim in tqdm(uusim_df.values):
            self.user_user_similarity[user1][user2] = sim
            self.user_user_similarity[user2][user1] = sim

        if verbose:
            print('Similarities loaded from {}'.format(self.similarities_filename))

        if self.log is not None:
            self.log.append('Similarities loaded from {}'.format(self.similarities_filename))     

    def getUserItemMatrix(self, verbose=False):
        # create user-item matrix dictionary
        self.userItemMatrix = {}

        if verbose:
            print('*'*10, 'Creating user-item matrix...', '*'*10)
            start_time = time.time()

        if self.log is not None:
            self.log.append('Creating user-item matrix initiated')

        for user in self.train_users['user_id']:
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

        if self.log is not None:
            self.log.append('User-item matrix created. ' + 'Time taken: {:.2f} seconds'.format(time.time() - start_time))

    def getUserPairSimilarity(self, user1, user2):
        '''
        OBSOLETE FUNCTION

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

        if self.log is not None:
            self.log.append('Creating user-user similarity matrix initiated')

        # create user-user similarity matrix dictionary
        self.user_user_similarity = {}

        if verbose:
            print('*'*10, 'Creating user-user similarity matrix...', '*'*10)
            start_time = time.time()

        for user1 in tqdm(self.train_users['user_id'].unique()):
            self.user_user_similarity.setdefault(user1, {})
            for user2 in self.train_users['user_id'].unique():
                if user1 != user2:
                    self.user_user_similarity[user1][user2] = self.similarity(self.userItemMatrix[user1], self.userItemMatrix[user2])
                
        if verbose:
            print('Time taken: {:.2f} seconds'.format(time.time() - start_time))
            print('User-user similarity matrix created.')
            print()
            if self.notification_level >= 2:
                balloon_tip('SAShA Detection', 'User-user similarity matrix created.')

        if self.log is not None:
            self.log.append('User-user similarity matrix created. ' + 'Time taken: {:.2f} seconds'.format(time.time() - start_time))

        if self.similarities_filename is not None:
            self.saveSimilarities()

    def getRecommendations(self, user_id, n_neighbors=10, verbose=False, log_this=False):

        if self.user_user_similarity is None:
            self.getUserUserSimilarity(verbose=verbose)

        if self.userItemMatrix is None:
            self.getUserItemMatrix(verbose=verbose)

        if verbose:
            print('*'*10, 'Getting recommendations for user {}...'.format(user_id), '*'*10)
            start_time = time.time()

        if self.log is not None and log_this:
            self.log.append('Getting recommendations for user {}...'.format(user_id))

        # get top k similar users
        #DEBUG NEEDED MAY BE -- >>>
        top_k_similar_users = sorted(self.user_user_similarity[user_id].items(), key=lambda x: x[1], reverse=True)[:n_neighbors]

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

        if self.log is not None and log_this:
            self.log.append('Recommendations for user {} are:'.format(user_id))
            self.log.append('{}'.format(items_to_recommend))

        return items_to_recommend
    
    def getRecommendationsForAllUsers(self, n_neighbors=10, verbose=False, output_filename='output/recommendations.csv', sep=',', top_n=None):

        if self.user_user_similarity is None:
            self.getUserUserSimilarity(verbose=verbose)

        if verbose:
            print('*'*10, 'Getting recommendations for all users...', '*'*10)
            start_time = time.time()

        if self.log is not None:
            self.log.append('Getting recommendations for all users...')
        
        self.recommendations = {}
        for user in tqdm(self.train_users['user_id']):
            self.recommendations[user] = self.getRecommendations(user, n_neighbors=n_neighbors)
        
        if verbose:
            print('Time taken: {:.2f} seconds'.format(time.time() - start_time))
            print('Recommendations for all users generated')
            print()
            if self.notification_level >= 1:
                balloon_tip('SAShA Detection', 'Recommendations for all users generated') 

        if self.log is not None:
            self.log.append('Recommendations for all users generated. ' + 'Time taken: {:.2f} seconds'.format(time.time() - start_time))

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

        if self.log is not None:
            self.log.append('Recommendations written to file: {}'.format(output_filename))
        
        return self.recommendations
    