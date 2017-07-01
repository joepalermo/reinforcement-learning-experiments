import sys
from os.path import abspath, join, dirname
# add the top level package to sys.path to access utilities
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

import time
from utilities import init_state_action_map, \
                      choose_epsilon_greedy_action, \
                      choose_greedy_action, \
                      generate_random_episode, \
                      generate_epsilon_greedy_episode
from Gridworld import Gridworld

def expected_q(state, q, epsilon):
    actions = q[state].keys()
    num_actions = len(actions)
    expl_prob = epsilon / num_actions
    greedy_action = choose_greedy_action(q, state)
    action_probs = [1 - epsilon + expl_prob if action == greedy_action else expl_prob for action in actions]
    return sum([action_prob * q[state][action] for (action, action_prob) in zip(actions, action_probs)])

def expected_sarsa(env, epsilon=0.1, alpha=0.5, gamma=1):
    q = init_state_action_map(env)
    for i in xrange(100):
        state = env.reset()
        done = False
        while not done:
            action = choose_epsilon_greedy_action(q, state, epsilon)
            (next_state, reward, done, _) = env.step(action)
            td_error = reward + gamma * expected_q(next_state, q, epsilon) - q[state][action]
            q[state][action] += alpha * td_error
            state = next_state
    return q

def main():
    #env = Gridworld(kings_moves=True, wind=[0,0,0,1,1,1,2,2,1,0], stochastic_wind=False)
    env = Gridworld(kings_moves=False)
    num_episodes = 1000

    # determine the baseline performance that results from taking random moves
    avg = sum([len(generate_random_episode(env)) for _ in range(num_episodes)]) / float(num_episodes)
    print "baseline random performance: " + str(avg)

    # learn q
    print "running expected sarsa..."
    q = expected_sarsa(env)
    print "expected sarsa complete"

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
