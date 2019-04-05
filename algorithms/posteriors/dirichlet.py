import numpy as np

class Dirichlet():
    def __init__(self, alpha, categories, times_to_init):
        self.alpha = alpha
        self.categories = categories
        self.times_to_init_ = times_to_init

    def times_to_init(self):
        return self.times_to_init_

    def alpha_posterior(self, rewards):
        count = np.zeros_like(self.categories)
        for r in rewards:
            category = self.categories.index(r)
            count[category] = count[category] + 1
        return count + self.alpha

    def sample_arm(self, reward, n_samples=1):
        p_list = np.random.dirichlet(self.alpha_posterior(reward), n_samples)
        return np.dot(p_list, self.categories)

    def mean(self,rewards):
        ap = self.alpha_posterior(rewards)
        return np.dot(self.categories, ap) / sum(ap)
