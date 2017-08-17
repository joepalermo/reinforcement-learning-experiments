import sys
from os.path import abspath, join, dirname
# add the top level package to sys.path to access them
sys.path.insert(1, abspath(join(dirname(__file__), '..')))
sys.path.insert(1, abspath(join(dirname(__file__), '../ch6')))
sys.path.insert(1, abspath(join(dirname(__file__), '../environments')))

import time
import random
from Qnet import Qnet
from GridworldChase import GridworldChase, state_encoder
from q_learning import q_learning
from utilities import init_state_action_map, \
                      choose_epsilon_greedy_action, \
                      estimate_performance, \
                      visualize_performance
import numpy as np


def main():
    # define hyperparameters
    num_episodes = 1000
    epsilon = 0.9
    gamma = 0.9
    alpha = 0.1

    # create an env
    env = GridworldChase(10, 10, p_goal_move=0.5, agent_random_start=True, goal_random_start=True)

    # init q and get baseline random performance
    q = init_state_action_map(env)

    estimate_performance(env, q, 1)

    # learn q
    print("running q-learning...")
    q = q_learning(env, q, epsilon=epsilon, alpha=alpha, gamma=gamma, num_episodes=num_episodes)
    print("q-learning complete")

    # determine post-training performance
    estimate_performance(env, q, 0.01)
    visualize_performance(env, q, delay=0.15)

if __name__ == '__main__':
    main()
