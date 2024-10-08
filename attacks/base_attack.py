# this file is used to generate random attack profiles
# this implementation of random attack is taken from the repository: https://github.com/merrafelice/Semantic-Aware-Shilling-Attacks.git 
# and modified to fit our needs

from abc import ABCMeta, abstractmethod
import config as cfg
import os

class BaseAttack:

    def __init__(self, data, r_max, r_min, attack_size_percentage = cfg.ATTACK_SIZE_PERCENTAGE, filler_size_percentage = cfg.FILLER_SIZE_PERCENTAGE, push = cfg.PUSH):
        """
        - data: the dataset
        - r_max: the maximum rating
        - r_min: the minimum rating
        - attack_size_percentage: the percentage of the dataset to be used for the attack
        - filler_size_percentage: the percentage of the dataset to be used for the filler
        - push: whether the attack is push or nuke
        """
        self.data = data
        self.r_max = r_max
        self.r_min = r_min
        self.target = push
        self.attackSizePercentage = attack_size_percentage
        self.fillerSizePercentage = filler_size_percentage
        self.targetRating = r_max if push else r_min
        self.fillerRating = int(r_max - r_min)
        self.datasetMean = self.data.rating.mean()
        self.datasetStd = self.data.rating.std()
        self.project_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))


    @abstractmethod
    def generate_profile(self, target_item_id, sample, output_filename): raise NotImplementedError

    @abstractmethod
    def get_filler_items(self, selected, target_item_id): raise NotImplementedError

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