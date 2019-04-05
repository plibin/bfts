import numpy as np
import math

class TDistribution():
    def __init__(self, alpha):
        self.alpha = alpha
        self.times_to_init_ = max(2, 3 - math.ceil(2*self.alpha))

    def freedom(self, n):
        return n + (2*self.alpha)-1

    def times_to_init(self):
        return self.times_to_init_

    def sample_arm(self,reward):
        n = len(reward)
        freedom = self.freedom(n)
        sigma = np.sqrt(np.var(reward)/freedom)
        mu = np.mean(reward)
        sample = np.random.standard_t(freedom)
        return mu + sample*sigma

    def mean(self, rewards):
        return np.mean(rewards)
