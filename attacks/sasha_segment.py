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


class SAShA_SegmentAttack(SemanticAttack):
    def __init__(self, 
                 data, 
                 r_max, 
                 r_min,  
                 similarity,
                 kg_item_feature_matrix,
                 similarity_filelocation,
                 attack_size_percentage=cfg.ATTACK_SIZE_PERCENTAGE, 
                 filler_size_percentage=cfg.FILLER_SIZE_PERCENTAGE, 
                 select_size_percentage=cfg.SELECT_SIZE_PERCENTAGE,
                 push=cfg.PUSH,
                 log=None):

        super(SAShA_SegmentAttack, self).__init__(data, 
                                                r_max, 
                                                r_min, 
                                                similarity,
                                                kg_item_feature_matrix,
                                                similarity_filelocation,
                                                attack_size_percentage, 
                                                filler_size_percentage, 
                                                push,
                                                log)
        
        self.select_size_percentage = select_size_percentage
        self.fillerSize = self.get_filler_size()
        self.selectedSize = self.get_selected_size()
        self.attackSize = self.get_attack_size()

    def generate_profile(self, target_items, sample, output_filename, verbose=False):

        """
        Generates the shilling profiles
        :param target_items: the target items
        :param sample: first fraction of the most and least similar items to be selected as selected and filler items respectively
        :param output_filename: the output filename, where shilling profiles are saved
        """

        start_shilling_user_id = max(list(self.data.user_id.unique()))
        shilling_profiles = []

        for target_item_id in (tqdm(target_items, leave=False) if verbose else target_items):
            for i in range(self.attackSize):
                start_shilling_user_id += 1
                # ADD SELECTED: First fraction of the most similar items to the target item, rated max
                selected_items = self.get_selected_items(target_item_id, sample)
                for selected_item_id in selected_items:
                    shilling_profiles.append([
                        start_shilling_user_id,
                        selected_item_id,
                        self.r_max
                    ])

                # ADD FILLER: First fraction of the least similar items to the target item, rated min
                filler_items = self.get_filler_items(selected_items, target_item_id, sample)
                for filler_item_id in filler_items:
                    shilling_profiles.append([
                        start_shilling_user_id,
                        filler_item_id,
                        self.r_min
                    ])

                # ADD TARGET ITEM with Rating (Max for Push/mn for Nuke)
                shilling_profiles.append([
                    start_shilling_user_id,
                    target_item_id,
                    self.targetRating
                ])


        # save shilling profiles
        shilling_profiles = pd.DataFrame(shilling_profiles, columns=['user_id', 'item_id', 'rating'])
        shilling_profiles.to_csv(output_filename, index=False)


    def get_filler_size(self):
        """
        Returns the size of the filler items
        """

        
        fillerSize = int((self.data.shape[0] / self.data.user_id.nunique() - 1)*self.fillerSizePercentage * (1 - self.select_size_percentage))

        return fillerSize
    
    def get_selected_size(self):
        """
        Returns the size of the selected items
        """
        
        selectedSize = int((self.data.shape[0] / self.data.user_id.nunique() - 1)*self.fillerSizePercentage * self.select_size_percentage)


        return selectedSize
    
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
        :param sample: first fraction of the least similar items to be selected as filler items
        :return: list of filler items RANDOMLY CHOSEN from sample
        """
        selectedItems.append(target_item_id)

        # Get least Similar Items
        similar_items = np.array(self.get_similar_items(target_item_id, sample, False))
        # Remove Selected Items
        
        similar_items = similar_items[~np.isin(similar_items, selectedItems)]
        items = random.choices(similar_items, k=self.fillerSize)

        return items
    
    def get_selected_items(self, target_item_id, sample):
        """
        randomly select from the items that are not in the selected items

        :param target_item_id: Target Item ID
        :param sample: first fraction of the most similar items to be selected as selected items
        :return: list of selected items RANDOMLY CHOSEN from sample
        """

        # Get least Similar Items
        similar_items = np.array(self.get_similar_items(target_item_id, sample))
        # Remove Selected Items
        
        similar_items = similar_items[~np.isin(similar_items, [target_item_id])]
        items = random.choices(similar_items, k=self.selectedSize)

        return items
    