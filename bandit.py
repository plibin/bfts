class Bandit:
    #arms: functions that represents the arm's reward distribution
    def __init__(self, arms):
        self.arms = arms

    def play(self,  arm_i):
        reward = self.arms[arm_i]()
        return reward
