import numpy as np
import pandas as pd
from utils.data_loader import *
from utils.similarity_measures import *
from tqdm import tqdm
import time

class MatrixFactorizationCF:
    def __init__(self, data, K, alpha, beta, iterations):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        Arguments
        - data (ndarray)   : user-item rating matrix
        - K (int)          : number of latent features
        - alpha (float)    : learning rate
        - beta (float)     : regularization parameter
        - iterations (int) : number of steps to take when
                             optimizing the W and H matrices
        """

        self.data = data
        self.num_users, self.num_items = data.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def train(self, verbose=False, show_mse=False):
        # Initialize user and item latent feature matrice
        self.W = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.H = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.data[np.where(self.data != 0)])

        # Create a list of training samples
        self.samples = [
            (i, j, self.data[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.data[i, j] > 0
        ]

        # Perform stochastic gradient descent for number of iterations
        training_process = []

        if verbose:
            print('*'*10, 'Starting training...', '*'*10)
            start_time = time.time()

        for i in tqdm(range(self.iterations)):
        # for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))

            if (i+1) % 10 == 0 and show_mse:
                print("Iteration: %d ; error = %.4f" % (i+1, mse))


        if verbose:
            print('*'*10, 'Training finished...', '*'*10)
            print('Total training time: ', time.time() - start_time)

        return training_process

    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.data.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.data[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

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
    
    def predict(self, i, j):
        """
        Predict the rating of user i for item j
        """
        return self.get_rating(i, j)
    
    def print_results(self):

        # Print results
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
        # df.to_csv(path, index=True, header=True)
        df.to_csv(path, index=False, header=False)

    
    def save_data_matrix(self, path):
        """
        Save the data matrix to a file
        """
        df = pd.DataFrame(self.data)
        # df.to_csv(path, index=True, header=True)
        df.to_csv(path, index=False, header=False)