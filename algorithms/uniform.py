import numpy as np

class Uniform:
    def __init__(self, bandit, m):
        self.bandit = bandit
        self.m = m

        self.rewards_per_arm = [[] for i in range(len(self.bandit.arms))]
        self.mean_per_arm = np.full(len(bandit.arms), float(0))

    def add_reward(self, arm_i, reward):
        self.rewards_per_arm[arm_i].append(reward)
        self.mean_per_arm[arm_i] = np.mean(self.rewards_per_arm[arm_i])

    def least_sampled_indices(self):
        count_per_arm = list(map(len, self.rewards_per_arm))
        return np.where(count_per_arm == np.min(count_per_arm))[0]

    def top_m(self):
        return np.argsort(-self.mean_per_arm)[0:self.m]

    def step(self, t):
        least_sampled = self.least_sampled_indices()
        arm_i = np.random.choice(least_sampled)
        reward = self.bandit.play(arm_i)
        self.add_reward(arm_i, reward)
        return self.top_m() 
