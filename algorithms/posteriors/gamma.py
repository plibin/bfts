import scipy.stats as stats
import numpy as np
import math

class Gamma():
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta 

    def times_to_init(self):
        #beta + n > 0
        if self.beta <= 0:
            return math.floor(-self.beta + 1)
        else: 
            return 0

    def sample_arm(self, rewards, n_samples=1):
        alpha_p = self.alpha + np.sum(rewards)
        beta_p = self.beta + len(rewards)
        return stats.gamma.rvs(a=alpha_p, scale=1/beta_p, size=n_samples)

    def mean(self,rewards):
        alpha_p = self.alpha + np.sum(rewards)
        beta_p = self.beta + len(rewards)
        return alpha_p / beta_p
