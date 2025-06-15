from bandit import Bandit

import numpy as np

import random

def bernoulli_means(n):
    means = [i / (n - 1) for i in range(n)]
    random.Random(1).shuffle(means)
    return means

def bernoulli_bandit(n):
    means = bernoulli_means(n) 
    
    def reward_fn(p_):
        return lambda: np.random.binomial(size=1, n=1, p=p_)
    arms = list(map(reward_fn, means))
    return Bandit(arms)
