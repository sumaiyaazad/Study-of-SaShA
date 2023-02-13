import numpy as np

# shilling attack random attack
class RandomAttack(object):
    def __init__(self, model, n_classes, n_features, n_samples, epsilon):
        self.model = model
        self.n_classes = n_classes
        self.n_features = n_features
        self.n_samples = n_samples
        self.epsilon = epsilon

    def attack(self, x, y):
        x_adv = np.zeros((self.n_samples, self.n_features))
        for i in range(self.n_samples):
            x_adv[i] = x[i] + np.random.uniform(-self.epsilon, self.epsilon, self.n_features)
        return x_adv
    
    def attack_batch(self, x, y, batch_size):
        x_adv = np.zeros((self.n_samples, self.n_features))
        for i in range(0, self.n_samples, batch_size):
            x_adv[i:i+batch_size] = x[i:i+batch_size] + np.random.uniform(-self.epsilon, self.epsilon, (batch_size, self.n_features))
        return x_adv
    
    