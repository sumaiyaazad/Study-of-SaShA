import pandas as pd
import numpy as np


class PredictionDifferenceDetector:
    def __init__(self, data):
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

        # calculate user and item means
        user_means = data_calc.groupby('user_id')['rating'].mean()
        item_means = data_calc.groupby('item_id')['rating'].mean()
        global_mean = data_calc['rating'].mean()

        # make predictions
        predictions = []

        for _, row in data_prediction.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']

            user_mean = user_means[user_id]
            item_mean = item_means[item_id]

            prediction = global_mean + (user_mean - global_mean) + (item_mean - global_mean)
            predictions.append(prediction)
        return predictions

    def predict_fake_profiles(self, fake_profiles_filename):
        """
        :param fake_profile_filename: filename to save fake profiles and npd values
        """
        npd_values = pd.DataFrame(columns=['user_id', 'npd'])
        for user_id in self.data['user_id'].unique():
            npd = self.calculate_npd(user_id)
            npd_values = npd_values.append({'user_id': user_id, 'npd': npd}, ignore_index=True)

        npd_mean = npd_values['npd'].mean()
        npd_std = npd_values['npd'].std()

        threshold = npd_mean + 3 * npd_std

        # filter fake profiles
        fake_profiles = npd_values[npd_values['npd'] > threshold]
        npd_values.to_csv(fake_profiles_filename, index=False)


        # npd_values = {user_id: self.calculate_npd(user_id) for user_id in self.data['user_id'].unique()}
        # mean = np.mean(list(npd_values.values()))
        # std = np.std(list(npd_values.values()))
        # threshold = mean + 3 * std
        # fake_profiles = [user_id for user_id, npd_value in npd_values.items() if npd_value > threshold]
        # fake_profiles_values = {user_id: 1 if npd_value > threshold else 0 for user_id, npd_value in npd_values.items()}


        # np.savetxt(npd_filename, np.array(list(npd_values.values())), fmt='%.2f')
        # np.savetxt(fake_profile_filename, np.array(list(fake_profiles_values.values())), fmt='%.0f')
        # np.savetxt(fake_profile_id, fake_profiles, fmt='%.0f')

        return fake_profiles
