from bandit import Bandit

import numpy as np
import pandas as pd

import random

def rewards_by_arm(csv_fn):
    df = pd.read_csv(csv_fn)
    return df.groupby('arm_index')['reward'].apply(list).to_dict()

def csv_dist_means(csv_fn):
    rewards_by_arm_ = rewards_by_arm(csv_fn)
    return [np.mean(rewards) for rewards in rewards_by_arm_.values()]

def csv_dist_bandit(csv_fn):
    rewards_by_arm_ = rewards_by_arm(csv_fn)
    def reward_fn(arm):
        return lambda: np.random.choice(rewards_by_arm_[arm])
    arms = list(map(reward_fn, rewards_by_arm_.keys()))
    return Bandit(arms)
