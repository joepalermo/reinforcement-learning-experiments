import random

class Arm:

    def __init__(self, max_random_step=0.1):
        self.true_value = random.gauss(0,1)
        self.max_random_step = max_random_step

    def __str__(self):
        return str(self.true_value)

    def get_reward(self):
        return random.gauss(self.true_value, 1)

    def random_step(self):
        self.true_value += random.gauss(0, self.max_random_step)

class Bandit:

    def __init__(self, k, initial_q_values=0, stationary=True):
        self.k = k
        self.arms = [Arm() for i in range(k)]

    def __str__(self):
        return 'true values: ' + ', '.join([str(arm) for arm in self.arms])

    def max_possible_expected_reward(self):
        return max([arm.true_value for arm in self.arms])

    def crank_arm(self, i):
        return self.arms[i].get_reward()
