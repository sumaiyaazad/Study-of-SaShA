from abc import ABCMeta, abstractmethod
from utils.similarity_measures import *
from utils.log import Logger
import config as cfg
import os
import pandas as pd
from tqdm import tqdm
import time

class SemanticAttack:

    def __init__(self,  data, 
                        r_max, 
                        r_min, 
                        similarity,
                        kg_item_feature_matrix,
                        similarity_filelocation,
                        attack_size_percentage = cfg.ATTACK_SIZE_PERCENTAGE, 
                        filler_size_percentage = cfg.FILLER_SIZE_PERCENTAGE, 
                        push = cfg.PUSH,
                        log = None):
        """
        - data: the dataset
        - r_max: the maximum rating
        - r_min: the minimum rating
        - similarity: the similarity meauser function
        - kg_item_feature_matrix: the kg item feature matrix (item_id, feature)
        - attack_size_percentage: the percentage of the dataset to be used for the attack
        - filler_size_percentage: the percentage of the dataset to be used for the filler
        - push: whether the attack is push or nuke
        - kg_items_filelocation: the location of the kg items file
                                    (feature, item_id)
                                    no header 
        - similarity_filelocation: the location of the similarity file (to save or load)
                                    (item_id, item_id, similarity)
                                    with header
        - log: the logger
        """
        self.data = data
        self.r_max = r_max
        self.r_min = r_min
        self.target = push
        self.targetRating = r_max if push else r_min
        self.fillerRating = int(r_max - r_min)
        self.datasetMean = self.data.rating.mean()
        self.datasetStd = self.data.rating.std()

        self.similarity = similarity
        self.similarity_filelocation = similarity_filelocation

        self.attackSizePercentage = attack_size_percentage
        self.fillerSizePercentage = filler_size_percentage
        self.project_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))

        self.item_item_similarity = None
        self.item_feature_matrix = kg_item_feature_matrix
        self.log = log

        self.load_similarity()

    def generate_similarity(self, verbose=False):
        """
        Generates the similarity matrix
        """

        if self.log is not None:
            self.log.append('Creating item-item similarity matrix initiated')
            start_time = time.time()

        # create item-item similarity matrix dictionary
        self.item_item_similarity = {}

        if verbose:
            print('*'*10, 'Creating item-item similarity matrix...', '*'*10)
            start_time = time.time()

        items = self.data['item_id'].unique().tolist()

        for item1 in tqdm(items):
            self.item_item_similarity.setdefault(item1, {})
            for item2 in items:
                if item1 != item2:
                    self.item_item_similarity[item1][item2] = self.similarity(self.item_feature_matrix[item1], self.item_feature_matrix[item2])

        if verbose:
            print('Time taken: {:.2f} seconds'.format(time.time() - start_time))
            print('Item-item similarity matrix created.')
            print()

        if self.log is not None:
            self.log.append('Item-item similarity matrix created. ' + 'Time taken: {:.2f} seconds'.format(time.time() - start_time))

        if self.similarity_filelocation is not None:
            self.save_similarity()

    def load_similarity(self, verbose=False):

        if self.log is not None:
            self.log.append('Loading kg similarities initiated')

        if self.similarity_filelocation is None:
            print('No filename given to save kg similarities')
            if self.log is not None:
                self.log.append('No filename given to save kg similarities')
                self.log.abort()
            raise ValueError('No filename given to load kg similarities')

        print('*'*10, 'Loading kg similarities...', '*'*10)

        # load as csv
        try:
            iisim_df = pd.read_csv(self.similarity_filelocation)
            iisim_df.columns = ['item1', 'item2', 'similarity']
        except FileNotFoundError:
            print('WARNING:File not found. Similarities will be calculated and saved to {}'.format(self.similarity_filelocation))
            if self.log is not None:
                self.log.append('WARNING:File not found. Similarities will be calculated and saved to {}'.format(self.similarity_filelocation))
            self.generate_similarity(verbose=verbose)
            return

        items = self.data['item_id'].unique().tolist()

        self.item_item_similarity = {}
        for item in items:
            self.item_item_similarity.setdefault(item, {})
            
        for item1, item2, sim in tqdm(iisim_df.values):
            self.item_item_similarity[item1][item2] = sim
            self.item_item_similarity[item2][item1] = sim

        if verbose:
            print('Kg Similarities loaded from {}'.format(self.similarity_filelocation))  

        if self.log is not None:
            self.log.append('Kg Similarities loaded from {}'.format(self.similarity_filelocation))      

    def save_similarity(self, verbose=False):
        '''
        Save item item similarities to file
        (item1, item2, similarity)
        '''

        if self.log is not None:
            self.log.append('saving similarities to {} initiated'.format(self.similarity_filelocation))

        if self.similarity_filelocation is None:
            print('No filename given to save similarities')
            if self.log is not None:
                self.log.append('No filename given to save similarities')
                self.log.abort()
            raise ValueError('No filename given to save similarities')

        if self.item_item_similarity is None:
            self.generate_similarity(verbose=verbose)

        print('*'*10, 'Saving similarities...', '*'*10)

        # convert to list of tuples
        iisim = []

        items = self.data['item_id'].unique().tolist()

        for item1 in tqdm(items):
            for item2 in items:
                if item1 != item2:
                    iisim.append([item1, item2, self.item_item_similarity[item1][item2]])

        # to dataframe
        iisim = pd.DataFrame(iisim, columns=['item1', 'item2', 'similarity'])

        # save as csv
        iisim.to_csv(self.similarity_filelocation, index=False)

        if self.log is not None:
            self.log.append('Similarities saved to {}'.format(self.similarity_filelocation))
        
        print('Similarities saved to {}'.format(self.similarity_filelocation))

    @abstractmethod
    def generate_profile(self, target_item_id, sample, output_filename): raise NotImplementedError

    @abstractmethod
    def get_filler_items(self, selected, target_item_id, sample): raise NotImplementedError

    @abstractmethod
    def get_selected_items(self, target_item_id): raise NotImplementedError

    @abstractmethod
    def get_filler_size(self): raise NotImplementedError

    @abstractmethod
    def get_selected_size(self): raise NotImplementedError

    def clamp(self, x):
        """
        :param x: the value to be clamped
        :return: the clamped value
        """
        return max(self.r_min, min(x, self.r_max))
    
    def get_similar_items(self, item_id, sample, verbose=False):
        """
        :param item_id: the item id to find similar items for
        :param sample: the sample to find similar items for

        :return: a list of similar items from {sample} quantile of the similarity distribution

        """

        if self.item_item_similarity is None:
            self.load_similarity(verbose=verbose)

        sim_df = pd.DataFrame(self.item_item_similarity[item_id].items(), columns=['item_id', 'similarity'])
        sim_df['item_id'] = sim_df['item_id'].astype(int)


        q1 = np.quantile(sim_df['similarity'], sample)
        similar = sim_df[sim_df['similarity'] >= q1]['item_id'].tolist()

        return similar