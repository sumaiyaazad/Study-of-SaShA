import random
from utils.data_loader import *
import time
from tqdm import tqdm
import numpy as np
import math as m
import pandas as pd
from utils.similarity_measures import *
import pickle


# implementation of item based collaborative filtering

class ItemBasedCF:

    def __init__(self, train_data, user_data, item_data, n_users=None, n_items=None, similarity=cosine_similarity):


        if n_users is not None and n_items is not None:
            self.train_data = train_data

            self.user_data = user_data
            self.item_data = item_data

            # pandas dataframe unique user and item
            self.n_users = len(self.user_data['user_id'].unique())
            self.n_items = len(self.item_data['item_id'].unique())

        elif n_users is not None and n_items is None:
            
            self.user_data = user_data[:n_users]
            self.item_data = item_data

            self.n_users = n_users
            self.n_items = len(self.item_data['item_id'].unique())
            
            self.train_data = train_data.loc[train_data['user_id'].isin(self.user_data['user_id']) & train_data['item_id'].isin(self.item_data['item_id'])]

        elif n_users is None and n_items is not None:
            
            self.user_data = user_data
            self.item_data = item_data[:n_items]
                
            self.n_users = len(self.user_data['user_id'].unique())
            self.n_items = n_items
            
            self.train_data = train_data.loc[train_data['user_id'].isin(self.user_data['user_id']) & train_data['item_id'].isin(self.item_data['item_id'])]
        else:

            self.n_users = n_users
            self.n_items = n_items
            
            self.user_data = user_data[:n_users]
            self.item_data = item_data[:n_items]
            
            self.train_data = train_data.loc[train_data['user_id'].isin(self.user_data['user_id']) & train_data['item_id'].isin(self.item_data['item_id'])]
            
        self.similarity = similarity
        self.itemUserMatrix = None
        self.item_item_similarity = None
        self.recommendations = None
        self.save_similarities = None

    def update_save_similarities(self, filename):
        self.save_similarities = filename

    def saveSimilarities(self):
        if self.save_similarities is None:
            print('No filename given to save similarities')
            return

        print('*'*10, 'Saving similarities...', '*'*10)

        # pickle dump
        with open(self.save_similarities, 'wb') as handle:
            pickle.dump(self.item_item_similarity, handle)

        
        # save as csv
        # pd.DataFrame(self.item_item_similarity).to_csv(self.save_similarities, index=False)
        
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
            self.item_item_similarity = pickle.load(handle)

        
        # load as csv
        # self.item_item_similarity = pd.read_csv(self.save_similarities).to_dict()

        print('Similarities loaded from {}'.format(self.save_similarities))            


    def getItemUserMatrix(self, verbose=False):
        # create item-user matrix dictionary
        self.itemUserMatrix = {}

        if verbose:
            print('*'*10, 'Creating item-user matrix...', '*'*10)
            start_time = time.time()

        for item in self.item_data['item_id']:
            self.itemUserMatrix.setdefault(item, {})

        for index, datapoint in tqdm(self.train_data.iterrows()):
            self.itemUserMatrix.setdefault(datapoint['item_id'], {})
            self.itemUserMatrix[datapoint['item_id']][datapoint['user_id']] = datapoint['rating']

        if verbose:
            print('Time taken: {:.2f} seconds'.format(time.time() - start_time))
            print('Item-user matrix created.')
            print()

    
    def getItemItemSimilarity(self, verbose=False):

        if self.itemUserMatrix is None:
            self.getItemUserMatrix(verbose=verbose)

        # create item-item similarity matrix dictionary
        self.item_item_similarity = {}

        if verbose:
            print('*'*10, 'Creating item-item similarity matrix...', '*'*10)
            start_time = time.time()

        for item1 in tqdm(self.item_data['item_id'].unique()):
            self.item_item_similarity.setdefault(item1, {})
            for item2 in self.item_data['item_id'].unique():
                if item1 != item2:
                    self.item_item_similarity[item1][item2] = self.similarity(self.itemUserMatrix[item1], self.itemUserMatrix[item2])

        if verbose:
            print('Time taken: {:.2f} seconds'.format(time.time() - start_time))
            print('Item-item similarity matrix created.')
            print()


        if self.save_similarities is not None:
            print('Saving similarities...')
            self.saveSimilarities()
            print('Similarities saved.')

    def getRecommendations(self, user_id, n_neighbors=10, verbose=False):

        if self.item_item_similarity is None:
            self.getItemItemSimilarity(verbose=verbose)

        if self.itemUserMatrix is None:
            self.getItemUserMatrix(verbose=verbose)

        if verbose:
            print('*'*10, 'Getting recommendations for user {}...'.format(user_id), '*'*10)
            start_time = time.time()

        # get unrated items by user
        rated_items = self.train_data[self.train_data['user_id'] == user_id]['item_id'].unique().tolist()
        unrated_items = set(self.item_data['item_id'].unique().tolist()).difference(rated_items)

        # filter rated items similarity
        iisim = {k: v for k, v in self.item_item_similarity.items() if k in rated_items}


        
        items_to_recommend = {}



### =================================================================================================== new code 0
        for item1 in unrated_items:
            
            top_k_similar_items = sorted(self.item_item_similarity[item1].items(), key=lambda x: x[1], reverse=True)[:n_neighbors]
            norm_factor = sum([x[1] for x in top_k_similar_items if x[0] in rated_items])
            for item2 in top_k_similar_items:
                if item2[0] in rated_items:
                    items_to_recommend.setdefault(item1, 0)
                    items_to_recommend[item1] += item2[1] * self.itemUserMatrix[item2[0]][user_id]
            if norm_factor != 0:
                items_to_recommend[item1] /= norm_factor


### =================================================================================================== new code 0

### =================================================================================================== new code
        # for item1 in unrated_items:
            
        #     # top_k_similar_items = sorted(self.item_item_similarity[item1].items(), key=lambda x: x[1], reverse=True)[:n_neighbors]
        #     norm_factor = sum([x[1] for x in self.item_item_similarity[item1].items() if x[0] in rated_items])
        #     for item2 in rated_items:
        #         items_to_recommend.setdefault(item1, 0)
        #         items_to_recommend[item1] += self.item_item_similarity[item1][item2] * self.itemUserMatrix[item2][user_id]

        #     if norm_factor != 0:
        #         items_to_recommend[item1] /= norm_factor


### =================================================================================================== new code

### =================================================================================================== old code
        # for item_id in unrated_items:
        #     # get top k similar items
            
        #     top_similar_items = sorted(self.item_item_similarity[item_id].items(), key=lambda x: x[1], reverse=True)

        #     # rated by user
        #     rated_by_user = top_similar_items.copy()
        #     for item in top_similar_items:
        #         if item[0] not in rated_items:
        #             rated_by_user.remove(item)


        #     top_k_similar_items = rated_by_user[:n_neighbors]
            
        #     items_to_recommend.setdefault(item_id, 0)
        #     norm_factor = sum([x[1] for x in top_k_similar_items])
        #     for similar_item in top_k_similar_items:
        #         items_to_recommend[item_id] += similar_item[1] * self.itemUserMatrix[similar_item[0]][user_id]

        #     if norm_factor != 0:
        #         items_to_recommend[item_id] /= norm_factor

### =================================================================================================== old code

        items_to_recommend = sorted(items_to_recommend.items(), key=lambda x: x[1], reverse=True)

        if verbose:
            print('Time taken: {:.2f} seconds'.format(time.time() - start_time))
            print('Recommendations for user {} obtained.'.format(user_id))
            print()


        return items_to_recommend
    
    def getRecommendationsForAllUsers(self, n_neighbors=10, verbose=False, output_filename='output/recommendations.csv', sep='::', top_n=None):

        if self.item_item_similarity is None:
            self.getItemItemSimilarity(verbose=verbose)

        if verbose:
            print('*'*10, 'Getting recommendations for all users...', '*'*10)
            start_time = time.time()
        
        self.recommendations = {}
        for user in tqdm(self.user_data['user_id']):
            self.recommendations[user] = self.getRecommendations(user, n_neighbors=n_neighbors)
        
        if verbose:
            print('Time taken: {:.2f} seconds'.format(time.time() - start_time))
            print('Recommendations for all users are:')
            print()

        # write to file
        
        
        # write the recommendation items to file
        if verbose:
            print("Write recommendations to file...")


        with open(output_filename, "w") as f:
            for user, items in self.recommendations.items():
                if top_n is not None:
                    # sort the items by rating
                    items = sorted(items, key=lambda x: x[1], reverse=True)
                    items = items[:top_n]
                for item, rating in items:
                    f.write(str(user) + sep + str(item) + sep + str(rating) + "\n")

        
        return self.recommendations
    