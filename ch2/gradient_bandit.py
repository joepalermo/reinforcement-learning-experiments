import sys
import random
import numpy as np
from os.path import abspath, join, dirname
# add the top level package to sys.path to access utilities and environments
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from utilities import softmax
from Bandit import Bandit

def choose_action(p):
    '''
    Choose an action using the probability distribution over actions. Each
    action is represented as the corresponding index into the probability
    distribution.
    '''
    return np.random.choice(list(range(len(p))), 1, p=p)[0]

def solve_bandit(bandit, timesteps=1000, step_size=0.01):
    '''
    Use the gradient-bandit method to try to extact the maximum reward from a
    k-arm bandit over a certain number of timesteps.
    '''
    n_steps = 0
    average_reward = 0
    utility = np.zeros(bandit.k)
    for _ in range(timesteps):
        policy = softmax(utility)
        arm_i = choose_action(policy)
        reward = bandit.crank_arm(arm_i)
        # update utility
        utility[arm_i] += step_size * (reward - average_reward) * (1 - policy[arm_i])
        for i in range(bandit.k):
            if i != arm_i:
                utility[i] += step_size * (reward - average_reward) * policy[i]
        # update average reward
        n_steps += 1
        average_reward += 1/n_steps * (reward - average_reward)
    reward_ratio = max(average_reward / bandit.max_possible_expected_reward() * 100, 0)
    return average_reward, reward_ratio

def solve_bandit_randomly(bandit, timesteps=1000):
    '''
    Choose random actions on a k-arm bandit for a certain number of timesteps
    keeping track of accumulated reward. Use for benchmarking against more
    intelligent methods.
    '''
    n_steps = 0
    average_reward = 0
    for _ in range(timesteps):
        arm_i = random.randint(0, bandit.k-1)
        reward = bandit.crank_arm(arm_i)
        # update average reward
        n_steps += 1
        average_reward += 1/n_steps * (reward - average_reward)
    reward_ratio = max(average_reward / bandit.max_possible_expected_reward() * 100, 0)
    return average_reward, reward_ratio

# observe some runs
bandit = Bandit(5)
print(bandit.max_possible_expected_reward())
print(solve_bandit_randomly(bandit))
print(solve_bandit(bandit))
print(solve_bandit(bandit))
print()
bandit = Bandit(5)
print(bandit.max_possible_expected_reward())
print(solve_bandit_randomly(bandit))
print(solve_bandit(bandit))
print(solve_bandit(bandit))
