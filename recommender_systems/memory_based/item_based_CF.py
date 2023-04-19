from utils.data_loader import *
from utils.similarity_measures import *
from utils.notification import *
import time
from tqdm import tqdm
import pandas as pd
from utils.log import Logger


class ItemBasedCF:

    def __init__(self, data, similarity_filename, similarity=cosine_similarity, notification_level=0, log=None):

        """
        Initialize the ItemBasedCF class
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
            log.append('Creating object of ItemBasedCF class')

        self.train_data, self.train_users, self.train_items = data

        self.n_users = len(self.train_users['user_id'].unique())
        self.n_items = len(self.train_items['item_id'].unique())
            
        self.similarity = similarity
        self.itemUserMatrix = None
        self.item_item_similarity = None
        self.recommendations = None
        self.similarities_filename = similarity_filename
        self.notification_level = notification_level
        self.log = log

        self.loadSimilarities()

    def update_similarities_filename(self, filename):
        self.similarities_filename = filename
        if self.log is not None:
            self.log.append('similarities filename updated to {}'.format(filename))

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

        if self.item_item_similarity is None:
            self.getItemItemSimilarity(verbose=True)

        print('*'*10, 'Saving similarities...', '*'*10)

        # convert to list of tuples
        iisim = []

        for item1 in tqdm(self.train_items['item_id'].unique()):
            for item2 in self.train_items['item_id'].unique():
                if item1 != item2:
                    iisim.append([item1, item2, self.item_item_similarity[item1][item2]])

        # to dataframe
        iisim = pd.DataFrame(iisim, columns=['item1', 'item2', 'similarity'])

        # save as csv
        iisim.to_csv(self.similarities_filename, index=False)

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
            iisim_df = pd.read_csv(self.similarities_filename)
            iisim_df.columns = ['item1', 'item2', 'similarity']
            # iisim_df = pd.read_csv(self.similarities_filename, header=None, names=['item1', 'item2', 'similarity'])
        except FileNotFoundError:
            print('WARNING:File not found. Similarities will be calculated and saved to {}'.format(self.similarities_filename))
            if self.log is not None:
                self.log.append('WARNING:File not found. Similarities will be calculated and saved to {}'.format(self.similarities_filename))
            self.getItemItemSimilarity(verbose=True)
            return

        self.item_item_similarity = {}
        for item in self.train_items['item_id'].unique():
            self.item_item_similarity.setdefault(item, {})
            
        for item1, item2, sim in tqdm(iisim_df.values):
            self.item_item_similarity[item1][item2] = sim
            self.item_item_similarity[item2][item1] = sim

        if verbose:
            print('Similarities loaded from {}'.format(self.similarities_filename))  

        if self.log is not None:
            self.log.append('Similarities loaded from {}'.format(self.similarities_filename))          

    def getItemUserMatrix(self, verbose=False):
        # create item-user matrix dictionary
        self.itemUserMatrix = {}

        if verbose:
            print('*'*10, 'Creating item-user matrix...', '*'*10)
            start_time = time.time()

        if self.log is not None:
            self.log.append('Creating item-user matrix initiated')
            start_time = time.time()

        for item in self.train_items['item_id']:
            self.itemUserMatrix.setdefault(item, {})

        for index, datapoint in tqdm(self.train_data.iterrows()):
            self.itemUserMatrix.setdefault(datapoint['item_id'], {})
            self.itemUserMatrix[datapoint['item_id']][datapoint['user_id']] = datapoint['rating']

        if verbose:
            print('Time taken: {:.2f} seconds'.format(time.time() - start_time))
            print('Item-user matrix created.')
            print()
            if self.notification_level >= 2:
                balloon_tip( 'SAShA Detection','Item-user matrix created.')

        if self.log is not None:
            self.log.append('Item-user matrix created. ' + 'Time taken: {:.2f} seconds'.format(time.time() - start_time))

    def getItemItemSimilarity(self, verbose=False):

        if self.itemUserMatrix is None:
            self.getItemUserMatrix(verbose=verbose)

        if self.log is not None:
            self.log.append('Creating item-item similarity matrix initiated')
            start_time = time.time()

        # create item-item similarity matrix dictionary
        self.item_item_similarity = {}

        if verbose:
            print('*'*10, 'Creating item-item similarity matrix...', '*'*10)
            start_time = time.time()

        for item1 in tqdm(self.train_items['item_id'].unique()):
            self.item_item_similarity.setdefault(item1, {})
            for item2 in self.train_items['item_id'].unique():
                if item1 != item2:
                    self.item_item_similarity[item1][item2] = self.similarity(self.itemUserMatrix[item1], self.itemUserMatrix[item2])

        if verbose:
            print('Time taken: {:.2f} seconds'.format(time.time() - start_time))
            print('Item-item similarity matrix created.')
            print()
            if self.notification_level >= 2:
                balloon_tip( 'SAShA Detection','Item-item similarity matrix created.')

        if self.log is not None:
            self.log.append('Item-item similarity matrix created. ' + 'Time taken: {:.2f} seconds'.format(time.time() - start_time))

        if self.similarities_filename is not None:
            self.saveSimilarities()

    def getRecommendations(self, user_id, n_neighbors=10, verbose=False, log_this=False):

        if self.item_item_similarity is None:
            self.getItemItemSimilarity(verbose=verbose)

        if self.itemUserMatrix is None:
            self.getItemUserMatrix(verbose=verbose)

        if verbose:
            print('*'*10, 'Getting recommendations for user {}...'.format(user_id), '*'*10)
            start_time = time.time()

        if self.log is not None and log_this:
            self.log.append('Getting recommendations for user {} initiated'.format(user_id))

        # get unrated items by user
        rated_items = self.train_data[self.train_data['user_id'] == user_id]['item_id'].unique().tolist()
        unrated_items = set(self.train_items['item_id'].unique().tolist()).difference(rated_items)
        
        items_to_recommend = {}

        for item1 in unrated_items:
            
            top_k_similar_items = sorted(self.item_item_similarity[item1].items(), key=lambda x: x[1], reverse=True)[:n_neighbors]
            norm_factor = sum([x[1] for x in top_k_similar_items if x[0] in rated_items])
            for item2 in top_k_similar_items:
                if item2[0] in rated_items:
                    items_to_recommend.setdefault(item1, 0)
                    items_to_recommend[item1] += item2[1] * self.itemUserMatrix[item2[0]][user_id]
            if norm_factor != 0:
                items_to_recommend[item1] /= norm_factor

        items_to_recommend = sorted(items_to_recommend.items(), key=lambda x: x[1], reverse=True)

        if verbose:
            print('Time taken: {:.2f} seconds'.format(time.time() - start_time))
            print('Recommendations for user {} obtained.'.format(user_id))
            print()
            if self.notification_level >= 2:
                balloon_tip( 'SAShA Detection','Recommendations for user {} obtained.'.format(user_id))

        if self.log is not None and log_this:
            self.log.append('Recommendations for user {} obtained'.format(user_id))
            
        return items_to_recommend
    
    def getRecommendationsForAllUsers(self, n_neighbors=10, verbose=False, output_filename='output/recommendations.csv', sep=',', top_n=None):

        if self.item_item_similarity is None:
            self.getItemItemSimilarity(verbose=verbose)

        if verbose:
            print('*'*10, 'Getting recommendations for all users...', '*'*10)
            start_time = time.time()

        if self.log is not None:
            self.log.append('Getting recommendations for all users initiated')
            start_time = time.time()
        
        self.recommendations = {}
        for user in tqdm(self.train_users['user_id']):
            self.recommendations[user] = self.getRecommendations(user, n_neighbors=n_neighbors)
        
        if verbose:
            print('Time taken: {:.2f} seconds'.format(time.time() - start_time))
            print('Recommendations for all users generated.')
            print()
            if self.notification_level >= 1:
                balloon_tip( 'SAShA Detection','Recommendations for all users generated.')

        if self.log is not None:
            self.log.append('Recommendations for all users generated. ' + 'Time taken: {:.2f} seconds'.format(time.time() - start_time))


        # write the recommendation items to file
        if verbose:
            print("Write recommendations to file...")

        if self.log is not None:
            self.log.append('Writing recommendations to file initiated')


        with open(output_filename, "w") as f:
            f.write("user_id" + sep + "item_id" + sep + "rating" + "\n")
            for user, items in self.recommendations.items():
                if top_n is not None:
                    # sort the items by rating (descending) and then by item id (ascending):    WILL IT WORK?
                    items = sorted(items, key=lambda x: (x[1],-x[0]), reverse=True)
                    # items = sorted(items, key=lambda x: x[1], reverse=True)
                    items = items[:top_n]
                for item, rating in items:
                    f.write(str(user) + sep + str(item) + sep + str(rating) + "\n")

        if self.log is not None:
            self.log.append('Recommendations written to file {}.'.format(output_filename))


        return self.recommendations
    