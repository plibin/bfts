import numpy as np
import sys

#AT-LUCB (Jun, ICML, 2016)
class AT_LUCB:
    def __init__(self, bandit, m, sigma1, alpha, epsilon):
        self.bandit = bandit
        self.m = m

        self.sigma1 = sigma1
        self.alpha = alpha
        self.epsilon = epsilon

        self.reward_per_arm = [[] for i in range(len(bandit.arms))]
        self.mean_per_arm = np.full(len(bandit.arms), float(0))

        self.Jt = np.full(self.m, -1)
        self.S = [1]

    def sigma(self, s):
        return self.sigma1 * self.alpha**(s-1)

    def beta(self, u, t, sigma1):
        k1 = 1.25
        n = len(self.bandit.arms)
        return ((np.log(n*k1*(t**4)/sigma1))/(2*u))**0.5

    def term(self, t, sigma, epsilon):
        h = self.h(t, sigma) #max
        l = self.l(t, sigma) #min
        U = self.U(t, l, sigma)
        L = self.L(t, h, sigma)
        return U - L < epsilon
        
    def L(self, t, a, sigma):
        mu = self.mean_per_arm[a]
        u = len(self.reward_per_arm[a])
        if u == 0:
            return float("-inf")
        else:
            return mu - self.beta(u, t, sigma)

    def U(self, t, a, sigma):
        mu = self.mean_per_arm[a]
        if mu == 0.0:
            return float("inf")
        else:
            return mu + self.beta(len(self.reward_per_arm[a]), t, sigma)

    def h(self, t, sigma):
        min_ = sys.float_info.max
        min_index = -1
        for j in self.Jt:
            L = self.L(t, j, sigma)
            if L < min_:
                min_ = L
                min_index = j

        return int(min_index)

    def l(self, t, sigma):
        max_ = sys.float_info.min
        max_index = -1
        for j in range(len(self.bandit.arms)):
            if j in self.Jt:
                pass
            else:
                U = self.U(t, j, sigma)
                if U > max_:
                    max_ = U
                    max_index = j
        
        return int(max_index)

    def top_m(self):
        return np.argsort(-self.mean_per_arm)[0:self.m]

    def add_reward(self, arm_i, reward):
        self.reward_per_arm[arm_i].append(reward)
        self.mean_per_arm[arm_i] = np.mean(self.reward_per_arm[arm_i])

    def step(self, t):
        if self.term(t, self.sigma(self.S[t - 1]), self.epsilon):
            s = self.S[t - 1]
            while self.term(t, self.sigma(s), self.epsilon):
                s = s + 1
            self.S.append(s)
            self.Jt = self.top_m()
        else:
            self.S.append(self.S[t-1])
            if self.S[t] == 1:
                self.Jt = self.top_m()

        lowest_index = self.l(t, self.sigma(self.S[t - 1]))
        low_reward = self.bandit.play(lowest_index)
        self.add_reward(lowest_index, low_reward)

        highest_index = self.h(t, self.sigma(self.S[t - 1]))
        high_reward = self.bandit.play(highest_index)
        self.add_reward(highest_index, high_reward)

        return self.Jt
