import pandas as pd
import numpy as np
from tqdm import tqdm


class PredictionDifferenceDetector:
    def __init__(self, data, constant):
        self.constant = constant
        if isinstance(data, pd.DataFrame):
            self.data = data
        else:
            self.data = pd.DataFrame(data)

    def calculate_npd(self, user_id):
        # remove user from data
        data_without_user = self.data[self.data['user_id'] != user_id]

        # make predictions with and without user
        predictions_with_user = self.make_predictions(self.data, data_without_user)
        predictions_without_user = self.make_predictions(data_without_user, data_without_user)

        npd = np.sum(np.abs(np.subtract(predictions_with_user, predictions_without_user)))

        return npd

    def make_predictions(self, data_calc, data_prediction):
        user_means = data_calc.groupby('user_id')['rating'].mean()
        item_means = data_calc.groupby('item_id')['rating'].mean()
        global_mean = data_calc['rating'].mean()

        user_mean = user_means.loc[data_prediction['user_id']].values
        item_mean = item_means.loc[data_prediction['item_id']].values

        prediction = global_mean + (user_mean - global_mean) + (item_mean - global_mean)
        return prediction

    def predict_fake_profiles(self, fake_profiles_filename):
        """
        :param fake_profile_filename: filename to save fake profiles and npd values
        """
        npd_values = pd.DataFrame(columns=['user_id', 'npd_value'])
        for user_id in tqdm(self.data['user_id'].unique(), leave=False):
            npd = self.calculate_npd(user_id)
            # print(user_id)
            npd_values.loc[len(npd_values)] = [user_id, npd]
        npd_values['user_id'] = npd_values['user_id'].astype(int)
        npd_mean = npd_values['npd_value'].mean()
        npd_std = npd_values['npd_value'].std()

        threshold = npd_mean + self.constant * npd_std

        # filter fake profiles
        fake_profiles = npd_values[npd_values['npd_value'] > threshold]
        npd_values.to_csv(fake_profiles_filename, index=False)

        return fake_profiles
