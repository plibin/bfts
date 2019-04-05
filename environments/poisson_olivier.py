from bandit import Bandit

import numpy as np

import random

#Cappé, Olivier, et al. "Kullback–leibler upper confidence bounds for optimal sequential allocation." The Annals of Statistics 41.3 (2013): 1516-1541.
def poisson_oli_means(n):
    min_mean = .5
    max_mean = 5
    fn = lambda i: min_mean + i * ((max_mean - min_mean) / (n-1))
    means = list(map(fn, range(n)))
    random.Random(1).shuffle(means)
    return means

def poisson_oli_bandit(n):
    means_ = poisson_oli_means(n)
    def reward_fn(lambda_):
        return lambda: np.random.poisson(lambda_)
    arms = list(map(reward_fn, means_))
    return Bandit(arms)
