import numpy as np
class Gaussian():
    def __init__(self, var, var0, mu0):
        self.known_var = var
        self.var0 = var0
        self.mu0 = mu0

    def times_to_init(self):
        return 0
    
    def sample_arm(self, rewards):
        n = len(rewards)
        denom = (1.0 / self.var0 + n / self.known_var)
        u = (self.mu0 / self.var0 + sum(rewards) / self.known_var) / denom
        var = 1.0 / denom
        stddev = np.sqrt(var)
        return np.random.normal(u, stddev)

    def mean(self, rewards):
        return np.mean(rewards)

