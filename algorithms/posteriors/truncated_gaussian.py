import numpy as np
from scipy import stats

class TruncatedGaussian():
    def __init__(self, a, b, var):
        self.a = a
        self.b = b
        self.known_var = var

    def times_to_init(self):
        return 1

    def sample_arm(self, rewards, n_samples=1):
        n = len(rewards)
        
        mu0 = np.mean(rewards)
        sigma0 = np.sqrt(self.known_var/n)
        a_norm = (self.a - mu0) / sigma0
        b_norm = (self.b - mu0) / sigma0
        
        return stats.truncnorm.rvs(a_norm, b_norm, loc=mu0, scale=sigma0, size=n_samples)
 
    def mean(self, rewards):
        n = len(rewards)

        if n == 0:
            return 0
        else:
            mu0 = np.mean(rewards)
            sigma0 = np.sqrt(self.known_var / n)
            a_norm = (self.a - mu0) / sigma0
            b_norm = (self.b - mu0) / sigma0
            return stats.truncnorm.mean(a_norm, b_norm, loc=mu0, scale=sigma0)
