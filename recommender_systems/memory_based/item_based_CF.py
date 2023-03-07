import numpy as np
import math as m
from operator import itemgetter
from similarity_measures import *
import random


# implementation of item based collaborative filtering
class ItemBasedCF:
    def __init__(self, train_file, test_file, n_users=100, n_items=1000, testing=False):
        self.train_file = train_file
        self.test_file = test_file
        self.userItemMatrix = None
        self.userSimilarity = None
        if testing:
            self.n_users = n_users
            self.n_items = n_items
            self.generateTestCases()

        self.readData()

    def readData(self):
        # Read data from file
        self.train = dict()
        for line in open(self.train_file):
            user, item, rating = line.strip().split("::")
            self.train.setdefault(user, {})
            self.train[user][item] = int(rating)

        self.test = dict()
        for line in open(self.test_file):
            user, item, rating = line.strip().split("::")
            self.test.setdefault(user, {})
            self.test[user][item] = int(rating)

        self.n_users = len(self.train)
        self.n_items = len(set([item for user in self.train for item in self.train[user]]))
        print("Number of users = %d, number of items = %d" % (self.n_users, self.n_items))

    
    def getUserItemMatrix(self):
        self.userItemMatrix = np.full((self.n_users, self.n_items), np.nan)

        for user in self.train.keys():
            for item in self.train[user].keys():
                self.userItemMatrix[int(user)][int(item)] = self.train[user][item]
        return self.userItemMatrix

    def getUserPairSimilarity(self, user1, user2):
        user1 = self.userItemMatrix[user1]
        user2 = self.userItemMatrix[user2]
        return cosine_similarity(user1, user2)
        

    def getUserSimilarity(self, simiarity=cosine_similarity):
        # calculate co-rated items between users

        
        if self.userItemMatrix is None:
            self.getUserItemMatrix()

        self.userSimilarity = np.zeros((self.n_users, self.n_users))
        for i in range(self.n_users):
            for j in range(self.n_users):
                self.userSimilarity[i][j] = simiarity(self.userItemMatrix[i], self.userItemMatrix[j])
        return self.userSimilarity
    
    def getItemSimilarity(self, simiarity=cosine_similarity):
        # calculate co-rated items between users
        if self.userItemMatrix is None:
            self.getUserItemMatrix()

        self.userSimilarity = np.zeros((self.n_users, self.n_users))
        for i in range(self.n_users):
            for j in range(self.n_users):
                self.userSimilarity[i][j] = simiarity(self.userItemMatrix[i], self.userItemMatrix[j])
        return self.userSimilarity
    

    def recommend(self, user, k=10):
        # find K nearest neighbors (users)
        nearest = np.argsort(self.userSimilarity[int(user)])[::-1][1:k+1]
        rank = dict()
        for item in self.train[user].keys():
            for j in nearest:
                if item in self.train[str(j)]:
                    # if the item is not rated by nearest neighbour user, then recommend it
                    rank.setdefault(item, 0)
                    rank[item] += self.userSimilarity[int(user)][j] * self.train[str(j)][item]
        return rank

    def evaluate(self, k=10):
        # return the recommendation items for each user
        self.getUserSimilarity()
        self.recommendations = dict()
        for user in self.train.keys():
            rank = self.recommend(user)
            self.recommendations[user] = dict(sorted(rank.items(), key=itemgetter(1), reverse=True)[0:k])

        # calculate precision and recall
        hit = 0
        precision = 0
        recall = 0
        for user in self.train.keys():
            # sort test items by rating in descending order
            test_recom = dict(sorted(self.test[user].items(), key=itemgetter(1), reverse=True)[:k])

            hit += set.intersection(set(self.recommendations[user].keys()), set(test_recom.keys())).__len__()
            precision += k
            recall += len(self.test[user])
        return (hit / (precision * 1.0)), (hit / (recall * 1.0))


    # generate the recommendation items for each user
    def generateRecommendations(self, k=10):
        print("Item-based CF recommendation system generate recommendations start...")
        self.getUserSimilarity()
        self.recommendations = dict()
        for user in self.train.keys():
            rank = self.recommend(user)
            self.recommendations[user] = dict(sorted(rank.items(), key=itemgetter(1), reverse=True)[0:k])

        # write the recommendation items to file
        print("Write recommendations to file...")
        with open("item_based_CF_recommendations.txt", "w") as f:
            for user, items in self.recommendations.items():
                for item, rating in items.items():
                    f.write(user + "::" + item + "::" + str(rating) + "")
    

    # generate test cases for the algorithm
    def generateTestCases(self):
        self.train = dict()
        self.test = dict()
        for i in range(self.n_users):
            self.train.setdefault(str(i), {})
            self.test.setdefault(str(i), {})
            for j in range(self.n_items):
                if random.random() < 0.8:
                    self.train[str(i)][str(j)] = random.randint(1, 5)
                else:
                    self.test[str(i)][str(j)] = random.randint(1, 5)


        # generate train data
        print("Write train data to file...")
        with open(self.train_file, "w") as f:
            for user, items in self.train.items():
                for item, rating in items.items():
                    f.write(user + "::" + item + "::" + str(rating) + "\n")


        # write the test cases to file
        print("Write test cases to file...")
        with open(self.test_file, "w") as f:
            for user, items in self.test.items():
                for item, rating in items.items():
                    f.write(user + "::" + item + "::" + str(rating) + "\n")
