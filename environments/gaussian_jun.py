from bandit import Bandit

import numpy as np

import random

def linear_means(n):
    mean_fn = lambda i: .9 * (n - i) / (n - 1)
    means = list(map(mean_fn, range(n)))
    random.Random(1).shuffle(means)
    return means

def linear_bandit(n, variance):
    means = linear_means(n)
    stddev = np.sqrt(variance)
    def reward_fn(mu):
        return lambda: np.random.normal(mu, stddev)
    arms = list(map(reward_fn, means))
    return Bandit(arms)

def polynomial_means(n):
    mean_fn = lambda i: .9 * (1 - np.sqrt(i / n))
    means = list(map(mean_fn, range(n)))
    random.Random(1).shuffle(means)
    return means

def polynomial_bandit(n, variance):
    means = polynomial_means(n)
    stddev = np.sqrt(variance)
    def reward_fn(mu):
        return lambda: np.random.normal(mu, stddev)
    arms = list(map(reward_fn, means))
    return Bandit(arms)
