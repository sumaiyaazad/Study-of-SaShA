import numpy as np
import pandas as pd
from utils.data_loader import *
from utils.similarity_measures import *
from tqdm import tqdm
import time
from utils.notification import *
from utils.log import Logger

class MatrixFactorizationCF:
    def __init__(self, data, users, items, K, alpha, beta, iterations, notification_level=0, log=None, r_min=1, r_max=5):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        Arguments
        - data (ndarray)   : user-item rating matrix (num_users x num_items), (user_index, item_index) = rating
        - K (int)          : number of latent features
        - alpha (float)    : learning rate
        - beta (float)     : regularization parameter
        - iterations (int) : number of steps to take when
                             optimizing the W and H matrices
        - users (dict)     : dictionary of users, maps user_id to index in the data matrix
        - items (dict)     : dictionary of items, maps item_id to index in the data matrix
        - notification_level (int) :    0 = no notifications, 
                                        1 = notify when recommendations are generated, 
                                        2 = notify when recommendations are generated and when training is complete
        """

        if log is not None:
            log.append('Creating object of MatrixFactorizationCF class')

        self.train_data = data
        self.train_users = users
        self.train_items = items
        self.num_users, self.num_items = len(users), len(items)
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.notification_level = notification_level
        self.log = log
        self.r_min = r_min
        self.r_max = r_max

    def train(self, verbose=False, show_mse=False):
        # Initialize user and item latent feature matrice
        self.W = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.H = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # Initialize the biases

        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.train_data[np.where(self.train_data != 0)])

        # Create a list of training samples
        # (user_index, item_index, rating)
        self.samples = [
            (i, j, self.train_data[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.train_data[i, j] > 0
        ]

        # Perform stochastic gradient descent for number of iterations
        training_process = []

        if verbose:
            print('*'*10, 'Starting training...', '*'*10)
            start_time = time.time()

        if self.log is not None:
            self.log.append('Starting training matrix factorization model')
            start_time = time.time()

        for i in tqdm(range(self.iterations)):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))

            if (i+1) % 10 == 0 and show_mse:
                print("Iteration: %d ; error = %.4f" % (i+1, mse))


        if verbose:
            print('*'*10, 'Training finished...', '*'*10)
            print('Total training time: ', time.time() - start_time)
            if self.notification_level >= 2:
                balloon_tip('SAShA Detection', 'Training finished')

        if self.log is not None:
            self.log.append('Finished training matrix factorization model. Total training time: ' + str(time.time() - start_time))

        return training_process

    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.train_data.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.train_data[x, y] - predicted[x, y], 2)
        return np.sqrt(error)
    
    def mae(self):
        """
        A function to compute the total mean absolute error
        """
        xs, ys = self.train_data.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += abs(self.train_data[x, y] - predicted[x, y])
        return error

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)

            # Update biases
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

            # Update user and item latent feature matrices
            self.W[i, :] += self.alpha * (2 * e * self.H[j, :] - self.beta * self.W[i,:])
            self.H[j, :] += self.alpha * (2 * e * self.W[i, :] - self.beta * self.H[j,:])

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.W[i, :].dot(self.H[j, :].T)
        return prediction
    
    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, W and H
        """
        return self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.W.dot(self.H.T)
    
    
    def print_results(self):
        print("P x Q:")
        print(self.full_matrix())
        print("Global bias:")
        print(self.b)
        print("User bias:")
        print(self.b_u)
        print("Item bias:")
        print(self.b_i)
        print("Left singular vectors:")
        print(self.W)
        print("Right singular vectors:")
        print(self.H)

    def get_recommendations(self, user_id, n=10):
        """
        Return the top n items with the highest predicted ratings
        """
        predictions = self.full_matrix()[user_id, :]
        items = np.argsort(predictions)[::-1][:n]
        return items
    
    def save_model(self, path):
        """
        Save the model to a file
        """
        np.savez(path, b=self.b, b_u=self.b_u, b_i=self.b_i, W=self.W, H=self.H)

    def load_model(self, path):
        """
        Load the model from a file
        """
        npzfile = np.load(path)
        self.b = npzfile["b"]
        self.b_u = npzfile["b_u"]
        self.b_i = npzfile["b_i"]
        self.W = npzfile["W"]
        self.H = npzfile["H"]

    def save_full_matrix(self, path):
        """
        Save the full matrix to a file
        """
        df = pd.DataFrame(self.full_matrix())
        df.to_csv(path, index=False, header=False)

    
    def save_data_matrix(self, path):
        """
        Save the data matrix to a file
        """
        df = pd.DataFrame(self.train_data)
        df.to_csv(path, index=False, header=False)
        
    def save_recommendations(self, output_path, n=10, verbose=False):
        """
        Save the recommendations to a file
        param output_path: the path to the output file
        param n: the number of recommendations to save
        param verbose: if True, print the progress
        """
        if verbose:
            print('*'*10, 'Saving recommendations...', '*'*10)
            start_time = time.time()

        if self.log is not None:
            self.log.append('Saving mfcf recommendations to ' + output_path)
            start_time = time.time()

        # exchange keys and values in the items dictionary
        items_rev = {v: k for k, v in self.train_items.items()}

        with open(output_path, "w") as f:
            f.write("user_id,item_id,rating\n")
            for user_id in tqdm(self.train_users.keys()):
                items = self.get_recommendations(self.train_users[user_id], n)
                for item in items:
                    f.write(str(user_id) + "," + str(items_rev[item]) + "," + str(self.clamp(self.get_rating(self.train_users[user_id], item))) + "\n")

        if verbose:
            print('*'*10, 'Recommendations saved...', '*'*10)
            print('Total saving time: ', time.time() - start_time)
            if self.notification_level >= 1:
                balloon_tip( 'SAShA Detection','Recommendations for all users generated.')

        if self.log is not None:
            self.log.append('Finished saving mfcf recommendations. Total saving time: ' + str(time.time() - start_time))


    def clamp(self, x):
        """
        :param x: the value to be clamped
        :return: the clamped value
        """
        return max(self.r_min, min(x, self.r_max))