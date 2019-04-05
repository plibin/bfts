import sys

import numpy as np

class BFTS:
    def __init__(self, bandit, m, posteriors):
        self.bandit = bandit
        self.m = m
        
        self.posteriors = posteriors
        
        self.rewards_per_arm = [[] for i in range(len(self.bandit.arms))]
        self.mean_per_arm = np.full(len(bandit.arms), float(0))

    def add_reward(self, arm_i, reward):
        self.rewards_per_arm[arm_i].append(reward)
        mean = self.posteriors[arm_i].mean(self.rewards_per_arm[arm_i])
        self.mean_per_arm[arm_i] = mean

    def top_m(self):
        n = len(self.bandit.arms)
        return np.argsort(-np.array(self.mean_per_arm))[0:self.m]

    def step(self, t):
        theta = np.zeros(len(self.bandit.arms))
        for i in range(len(self.bandit.arms)):
            rewards = self.rewards_per_arm[i]
            theta[i] = self.posteriors[i].sample_arm(rewards)
            
        order = np.argsort(-theta) 
        arm_i = order[self.m - 1 + np.random.choice([0,1])] 
        
        reward = self.bandit.play(arm_i)
        self.add_reward(arm_i, reward)

        return (self.top_m(), arm_i, reward)
