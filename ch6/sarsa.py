import sys
from os.path import abspath, join, dirname
# add the top level package to sys.path to access utilities and environments
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
sys.path.insert(1, abspath(join(dirname(__file__), '../environments')))

import time
from utilities import init_state_action_map, \
                      choose_epsilon_greedy_action, \
                      choose_greedy_action, \
                      generate_random_episode, \
                      generate_epsilon_greedy_episode
from Gridworld import Gridworld

def sarsa(env, epsilon=0.1, alpha=0.5, gamma=1):
    # learn q
    q = init_state_action_map(env)
    for i in xrange(1000):
        state = env.reset()
        action = choose_epsilon_greedy_action(q, state, epsilon)
        done = False
        while not done:
            (next_state, reward, done, _) = env.step(action)
            next_action = choose_epsilon_greedy_action(q, next_state, epsilon)
            td_error = reward + gamma * q[next_state][next_action] - q[state][action]
            q[state][action] += alpha * td_error
            state, action = next_state, next_action
    return q

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
    print "running sarsa..."
    q = sarsa(env)
    print "sarsa complete"

    # determine post-training performance
    avg = sum([len(generate_epsilon_greedy_episode(env, q)) for _ in range(num_episodes)]) / float(num_episodes)
    print "post learning performance: " + str(avg)

    # visualize post-training episode
    state = env.reset()
    while True:
        env.render()
        time.sleep(0.25)
        action = choose_greedy_action(q, state)
        state, _, done, _ = env.step(action) # take a random action
        if done:
            env.render(close=True)
            break

main()
