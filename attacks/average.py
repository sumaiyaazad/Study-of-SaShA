import numpy as np
from attacks.base_attack import BaseAttack
import config as cfg
import pandas as pd
from tqdm import tqdm
import os
import random


# Seed For Reproducibility
np.random.seed(cfg.SEED)
random.seed(cfg.SEED)


# shilling attack average attack
class AverageAttack(BaseAttack):
    def __init__(self, data, r_max, r_min, attack_size_percentage=cfg.ATTACK_SIZE_PERCENTAGE, filler_size_percentage=cfg.FILLER_SIZE_PERCENTAGE, push=cfg.PUSH):
        # drop colums timestamp
        if 'timestamp' in data.columns:
            data = data.drop(columns=['timestamp'])

        super(AverageAttack, self).__init__(data, r_max, r_min, attack_size_percentage, filler_size_percentage, push)
        self.fillerSize = self.get_filler_size()
        self.selectedSize = self.get_selected_size()
        self.attackSize = self.get_attack_size()

    def get_attack_size(self):
        """
        :return: The number of fake Profiles to be added (A Percentage of The Users in The Data Sample)
        """
        attackSize = int(self.data.user_id.nunique() * self.attackSizePercentage)
        return attackSize
    
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
        :return: The number of selected items to be added (A Percentage of The Items in The Data Sample)
        """
        return 0

    def get_filler_items(self, selectedItems, target_item_id):
        """
        randomly select from the items that are not in the selected items

        :param target_item_id: Target Item ID
        :param selectedItems: List of Already Selected Items
        :return: list of filler items RANDOMLY CHOSEN
        """
        selectedItems.append(target_item_id)

        items = self.data.item_id.unique()
        items = items[~np.isin(items, selectedItems)]
        items = random.choices(items, k=self.fillerSize)

        return items

    def get_selected_items(self, target_item_id):
        """
        no selected items required for random attack
        :return: List of Selected Items: EMPTY
        """
        return []

    def generate_profile(self, target_item_id, sample, output_filename):

        # mean of the ratings in the dataset for items
        items_mean = self.data.groupby('item_id')['rating'].mean()
        items_mean.name = 'mean_ratings'

        start_shilling_user_id = max(list(self.data.user_id.unique()))
        shilling_profiles = []

        for i in tqdm(range(self.attackSize)):
            start_shilling_user_id += 1

            # ADD SELECTED: Will Be Empty
            selected_items = self.get_selected_items(target_item_id)

            # ADD FILLER:   AVERAGE: Mean rating of the filler items in the system
            filler_items = self.get_filler_items(selected_items, target_item_id)
            for filler_item_id in filler_items:
                shilling_profiles.append([
                    start_shilling_user_id,
                    filler_item_id,
                    self.clamp(int(items_mean.loc[filler_item_id]))
                ])

            # ADD TARGET ITEM with Rating (Max for Push/mn for Nuke)
            shilling_profiles.append([
                start_shilling_user_id,
                target_item_id,
                self.targetRating
            ])

        shilling_profiles_df = pd.DataFrame(shilling_profiles, columns=list(self.data.columns))

        # Save File Of Shilling Profile in the Directory shilling_profiles
        # output_filename = "sample_{0}_{1}.csv".format(sample, int(target_item_id))

        shilling_profiles_df.to_csv(output_filename, index=False)

        return target_item_id
    
    def generate_profiles(self, target_item_ids, sample, output_dir):
        """
        Generate Shilling Profiles for a list of target items
        :param target_item_ids: List of Target Item IDs
        :param sample: Sample Number
        :param output_dir: Output Directory
        :return: List of Target Item IDs
        """
        for target_item_id in target_item_ids:
            output_filename = os.path.join(output_dir, "sample_{0}_{1}.csv".format(sample, int(target_item_id)))
            self.generate_profile(target_item_id, sample, output_filename)

        return target_item_ids
    
