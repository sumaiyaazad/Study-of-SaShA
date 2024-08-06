import pandas as pd
from scipy.stats import pearsonr
import numpy as np

class RDMADetector:
    def __init__(self, data, threshold):
        self.threshold = threshold
        self.df = pd.DataFrame(data)
        self.user_ids = self.df['user_id'].unique()
        self.item_ids = self.df['item_id'].unique()
        self.rdma = {}
        self.avg_sim = {}
        self.ps = {}
        self.shilling_profiles = []

    def calculate_avg_sim(self):
        user_item_matrix = self.df.pivot_table(index='user_id', columns='item_id', values='rating')
        similarity_matrix = user_item_matrix.corr(method='pearson', min_periods=2)
        similarity_matrix = similarity_matrix.fillna(0)
        top_25_neighbors = np.argsort(similarity_matrix.values)[:, -25:]
        avg_sim = np.zeros(len(similarity_matrix))
        for i in range(len(similarity_matrix)):
            avg_sim[i] = np.mean(similarity_matrix.values[i][top_25_neighbors[i]])
        self.avg_sim = dict(zip(similarity_matrix.index, avg_sim))

    def calculate_rdma(self):
        self.calculate_avg_sim()
        max_avg_sim = max(self.avg_sim.values())
        rdma_sum = 0
        for item in self.item_ids:
            item_df = self.df[self.df['item_id'] == item]
            mask = item_df['user_id'].isin([user for user, sim in self.avg_sim.items() if sim < max_avg_sim / 2])
            item_df = item_df[mask]
            NR = len(item_df)
            if NR == 0:
                continue
            avg_item_rating = item_df['rating'].mean()
            rdma_sum += (item_df['rating'] - avg_item_rating).abs() / NR
        rdma_sum /= len(self.user_ids)
        self.rdma = dict(zip(rdma_sum.index, rdma_sum.values))

    def calculate_ps(self):
        self.calculate_rdma()
        avg_rdma = np.mean(list(self.rdma.values()))
        for user in self.user_ids:
            rdma_value = self.rdma.get(user, 0)
            print(rdma_value)
            if rdma_value < avg_rdma:
                ps_u = 0
                # print(ps_u)
            else:
                ps_u = (np.exp(10 * (rdma_value - avg_rdma) / (1 - avg_rdma)) - 1) / (np.exp(10) - 1)
                # print(ps_u)
            self.ps[user] = ps_u

    def detect_shilling_profiles(self):
        self.calculate_ps()
        # self.calculate_rdma()
        for user in self.user_ids:
            ps = self.get_ps(user)
            # ps = self.rdma.get(user,0)
            if ps > self.threshold:
                self.shilling_profiles.append(user)

    def get_rdma(self, user):
        return self.rdma[user]

    def get_avg_sim(self, user):
        return self.avg_sim[user]

    def get_ps(self, user):
        return self.ps[user]

    def predict_fake_profiles(self, filename):
        # return self.ps
        self.detect_shilling_profiles()
        return self.shilling_profiles
