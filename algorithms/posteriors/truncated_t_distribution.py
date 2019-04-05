import numpy as np
import math
import scipy
from scipy.stats import t
from scipy.stats import uniform

from algorithms.posteriors.t_distribution import TDistribution

class TruncatedTDistribution():
    def __init__(self, alpha, a, b):
        self.t_dist = TDistribution(alpha)
        self.a = a
        self.b = b

    def times_to_init(self):
        return self.t_dist.times_to_init()

    def sample_arm(self,rewards, n_samples=1):
        #inverse transform sampling
        n = len(rewards)
        v = self.t_dist.freedom(n)
        mu = np.mean(rewards)
        sigma = np.sqrt(np.var(rewards)/self.t_dist.freedom(n))
        
        t_dist = t(df=v, loc=mu, scale=sigma)
        Fa = t_dist.cdf(self.a)
        Fb = t_dist.cdf(self.b)
        u = uniform.rvs(Fa, Fb-Fa, size=n_samples)
        return t_dist.ppf(u)

    def truncated_t_mean(self,rewards,mu,sigma):
        n = len(rewards)
        v = self.t_dist.freedom(n)
        #primitive function of the indefinite integral of the x*pdf(x), 
        #where pdf(x) is the pdf of the t-distribution
        def F(x):
            constant_n = scipy.special.gammaln((v+1)/2)
            constant_d = scipy.special.gammaln(v/2)
            constant = np.exp(constant_n - constant_d) / math.sqrt(v*math.pi)
            return constant * (v/(1-v) * (1 + x*x/v)**((1-v)/2))
        a_ = (self.a - mu)/sigma
        b_ = (self.b - mu)/sigma
        return (F(b_) - F(a_))/(t.cdf(b_, v)-t.cdf(a_, v))

    def mean(self, rewards):
        n = len(rewards)
        mu = np.mean(rewards)
        if n < self.times_to_init():
            return mu 
        else:
            sigma = np.sqrt(np.var(rewards)/self.t_dist.freedom(n))
            m = self.truncated_t_mean(rewards, mu, sigma)
            return m*sigma + mu
