import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


class PCAShillingAttackDetector:
    def __init__(self, data, n_components=10):
        self.data = data
        self.n_components = n_components
        self.user_item_matrix = self._construct_user_item_matrix()

    def _construct_user_item_matrix(self):
        # Create a pivot table to construct the user-item matrix
        user_item_matrix = self.data.pivot_table(index='user_id', columns='item_id', values='rating')

        # Fill missing values with zeros
        user_item_matrix = user_item_matrix.fillna(0)

        # Convert the pivot table to a numpy array
        user_item_matrix = user_item_matrix.to_numpy()

        return user_item_matrix

    def predict_fake_profiles(self, fake_profiles_filename):
        # Perform PCA on the user-item matrix
        pca = PCA(n_components=self.n_components)
        pca.fit(self.user_item_matrix)

        # Calculate the explained variance ratio of each principal component
        explained_variance_ratio = pca.explained_variance_ratio_

        # Find the principal component with the largest explained variance ratio
        max_component_index = np.argmax(explained_variance_ratio)

        # Project the user-item matrix onto the principal component with the largest explained variance ratio
        projected_matrix = pca.transform(self.user_item_matrix)[:, max_component_index]

        abs_projected_matrix = np.abs(projected_matrix)
        user_projected_matrix = {'user_id': self.data['user_id'].unique(), 'abs_value': abs_projected_matrix}
        user_projected_value = pd.DataFrame(user_projected_matrix)

        # Find the users with the largest absolute values in the projected matrix
        # These users are potential shilling attackers
        # you may choose the number of attackers
        # shilling_attackers = np.argsort(np.abs(projected_matrix))[-self.n_components:]
        shilling_attackers = user_projected_value.sort_values(by='abs_value', ascending=False)
        shilling_attackers = shilling_attackers.head(self.n_components).sort_values(by='user_id')

        shilling_attackers.to_csv(fake_profiles_filename, index=False)

        return shilling_attackers
