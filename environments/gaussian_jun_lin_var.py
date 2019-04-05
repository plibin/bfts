from bandit import Bandit
from environments.gaussian_jun import linear_means, polynomial_means

import numpy as np

def variances(n, max_variance, min_variance):
    fn = lambda i: max_variance - i*((max_variance - min_variance) / (n-1))
    return list(map(fn, range(n)))

def create_bandit(means, variances):
    def reward_fn(mu, variance):
        stddev = np.sqrt(variance)
        return lambda: np.random.normal(mu, stddev)
    arms = list(map(lambda a:reward_fn(*a), zip(means, variances)))
    return Bandit(arms)

def linear_bandit(n, max_variance, min_variance):
    means = linear_means(n)
    variances_ = variances(n, max_variance, min_variance)
    return create_bandit(means, variances_)

def polynomial_bandit(n, max_variance, min_variance):
    means = polynomial_means(n)
    variances_ = variances(n, max_variance, min_variance)
    return create_bandit(means, variances_)
