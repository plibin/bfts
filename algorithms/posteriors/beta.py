import numpy as np

#Beta posterior, resuling from a conjugate Beta prior
class BetaPosterior:
    #Initialise with hyperparameters to the Beta prior,
    #by default hyperparameters to the Jeffreys prior are chosen
    def __init__(self, alpha=0.5, beta=0.5):
        self.alpha = alpha
        self.beta = beta
    
    def times_to_init(self):
        return 0
    
    def successes_and_failures(self, rewards):
        successes = rewards.count(1)
        failures = rewards.count(0)

        #ensure that we have a list of binary rewards
        assert(successes == sum(rewards))
        
        return successes, failures 
    
    def mean(self,rewards):
        successes, failures = self.successes_and_failures(rewards) 

        alpha_p = self.alpha + successes
        beta_p = self.beta + failures

        return alpha_p / (alpha_p + beta_p)
    
    def sample_arm(self,rewards,n_samples=1):
        successes, failures = self.successes_and_failures(rewards) 

        alpha_p = self.alpha + successes
        beta_p = self.beta + failures

        return np.random.beta(alpha_p, beta_p, size=n_samples)
