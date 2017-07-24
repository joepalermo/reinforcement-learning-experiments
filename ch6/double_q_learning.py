import sys
import random
from os.path import abspath, join, dirname
# add the top level package to sys.path to access utilities and environments
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
sys.path.insert(1, abspath(join(dirname(__file__), '../environments')))

import time
from utilities import init_state_action_map, \
                      choose_greedy_action, \
                      generate_random_episode, \
                      generate_epsilon_greedy_episode
from Gridworld import Gridworld

def choose_doubled_epsilon_greedy_action(q1, q2, state, epsilon):
    actions = q1[state].keys()
    if random.random() < epsilon:
        return random.choice(actions)
    else:
        best_action_properties = (-1, -float("inf"))
        for i, action in enumerate(actions):
            action_value = q1[state][action] + q2[state][action]
            if action_value > best_action_properties[1]:
                best_action_properties = (i, action_value)
        best_action_i = best_action_properties[0]
        best_action = actions[best_action_i]
        return best_action

def double_q_update(q, q_prime, state, action, reward, next_state, alpha, gamma):
    maximizing_action = choose_greedy_action(q, next_state)
    td_error = reward + gamma * q_prime[next_state][maximizing_action] - q[state][action]
    q[state][action] = q[state][action] + alpha * td_error

def double_q_learning(env, epsilon=0.1, alpha=0.5, gamma=1):
    q1 = init_state_action_map(env)
    q2 = init_state_action_map(env)
    for i in xrange(100000):
        state = env.reset()
        done = False
        while not done:
            action = choose_doubled_epsilon_greedy_action(q1, q2, state, epsilon)
            (next_state, reward, done, _) = env.step(action)
            if random.random() < 0.5:
                double_q_update(q1, q2, state, action, reward, next_state, alpha, gamma)
            else:
                double_q_update(q2, q1, state, action, reward, next_state, alpha, gamma)
            state = next_state
    return q1, q2

def main():
    x_limit = 8
    y_limit = 4
    goals = [(7,0)]
    anti_goals = [(1,0),(2,0),(3,0),(4,0),(5,0),(6,0)]

    #env = Gridworld(x_limit, y_limit, goals, anti_goals, kings_moves=True, wind=[0,0,0,1,1,1,2,2,1,0], stochastic_wind=False)
    env = Gridworld(x_limit, y_limit, goals, anti_goals, kings_moves=False)
    num_episodes = 1000

    # determine the baseline performance that results from taking random moves
    avg = sum([len(generate_random_episode(env)) for _ in range(num_episodes)]) / float(num_episodes)
    print "baseline random performance: " + str(avg)

    # learn q
    print "running double q-learning..."
    q1, q2 = double_q_learning(env)
    print "double q-learning complete"

    # determine post-training performance
    avg = sum([len(generate_epsilon_greedy_episode(env, q1)) for _ in range(num_episodes)]) / float(num_episodes)
    print "post learning performance: " + str(avg)

    # visualize post-training episode
    state = env.reset()
    while True:
        env.render()
        time.sleep(0.25)
        action = choose_greedy_action(q2, state)
        state, _, done, _ = env.step(action) # take a random action
        if done:
            env.render(close=True)
            break

main()
