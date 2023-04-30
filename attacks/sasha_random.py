import numpy as np
from attacks.semantic_attack import SemanticAttack
import config as cfg
import pandas as pd
from tqdm import tqdm
import os
import random
from utils.log import Logger

np.random.seed(cfg.SEED)
random.seed(cfg.SEED)


class SAShA_RandomAttack(SemanticAttack):
    def __init__(self, 
                 data, 
                 r_max, 
                 r_min,  
                 similarity,
                 kg_item_feature_matrix,
                 similarity_filelocation,
                 attack_size_percentage=cfg.ATTACK_SIZE_PERCENTAGE, 
                 filler_size_percentage=cfg.FILLER_SIZE_PERCENTAGE, 
                 push=cfg.PUSH,
                 log=None):
        # drop colums timestamp
        if 'timestamp' in data.columns:
            data = data.drop(columns=['timestamp'])

        super(SAShA_RandomAttack, self).__init__(data, 
                                                r_max, 
                                                r_min, 
                                                similarity,
                                                kg_item_feature_matrix,
                                                similarity_filelocation,
                                                attack_size_percentage, 
                                                filler_size_percentage, 
                                                push,
                                                log)
        
        
        self.fillerSize = self.get_filler_size()
        self.selectedSize = self.get_selected_size()
        self.attackSize = self.get_attack_size()

    def generate_profile(self, target_items, sample, output_filename, verbose=False):

        """
        Generates the shilling profiles
        :param target_items: the target items
        :param sample: first fraction of the most similar items to be selected as filler items
        :param output_filename: the output filename, where shilling profiles are saved
        """

        start_shilling_user_id = max(list(self.data.user_id.unique()))
        shilling_profiles = []

        for target_item_id in (tqdm(target_items, leave=False) if verbose else target_items):
            for i in range(self.attackSize):
                start_shilling_user_id += 1

                # ADD SELECTED: Will Be Empty
                selected_items = self.get_selected_items(target_item_id)

                # ADD FILLER:   Random: Mean and Variance of dataset
                filler_items = self.get_filler_items(selected_items, target_item_id, sample)
                for filler_item_id in filler_items:
                    shilling_profiles.append([
                        start_shilling_user_id,
                        filler_item_id,
                        self.clamp(int(np.random.normal(self.datasetMean, self.datasetStd, 1).round()[0]))
                    ])

                # ADD TARGET ITEM with Rating (Max for Push/mn for Nuke)
                shilling_profiles.append([
                    start_shilling_user_id,
                    target_item_id,
                    self.targetRating
                ])

        shilling_profiles_df = pd.DataFrame(shilling_profiles, columns=list(self.data.columns))

        # Save File Of Shilling Profile in the Directory shilling_profiles
        shilling_profiles_df.to_csv(output_filename, index=False)

    def get_filler_size(self):
        """
        average number of items rated by users in the dataset
        |I_{F}|= #_of_all_ratings/|U| - 1
        :return: Filler Size
        """
        fillerSize = int((self.data.shape[0] / self.data.user_id.nunique() - 1)*self.fillerSizePercentage)

        return fillerSize

    def get_selected_size(self):
        """
        |I_{S}|= 0
        :return: Selected Size
        """
        selectedsize = 0
        return selectedsize

    def get_attack_size(self):
        """
        :return: The number of fake Profiles to be added (A Percentage of The Users in The Data Sample)
        """
        attackSize = int(self.data.user_id.nunique() * self.attackSizePercentage)
        return attackSize

    def get_filler_items(self, selectedItems, target_item_id, sample):
        """
        randomly select from the items that are not in the selected items

        :param target_item_id: Target Item ID
        :param selectedItems: List of Already Selected Items
        :param sample: first fraction of the most similar items to be selected as filler items
        :return: list of filler items RANDOMLY CHOSEN from sample
        """
        selectedItems.append(target_item_id)

        # Get Similar Items
        similar_items = np.array(self.get_similar_items(target_item_id, sample))
        # Remove Selected Items
        # similar_items = np.setdiff1d(similar_items, selectedItems)
        
        similar_items = similar_items[~np.isin(similar_items, selectedItems)]
        items = random.choices(similar_items, k=self.fillerSize)

        return items

    def get_selected_items(self, target_item_id):
        """
        no selected items required for random attack
        :return: List of Selected Items: EMPTY
        """
        return []
