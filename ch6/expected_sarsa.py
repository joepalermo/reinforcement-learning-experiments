import sys
from os.path import abspath, join, dirname
# add the top level package to sys.path to access utilities and environments
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
sys.path.insert(1, abspath(join(dirname(__file__), '../environments')))

from utilities import init_state_action_map, \
                      choose_epsilon_greedy_action, \
                      choose_greedy_action, \
                      generate_random_episode, \
                      generate_epsilon_greedy_episode, \
                      estimate_performance, \
                      visualize_performance
from Gridworld import Gridworld

def expected_q(state, q, epsilon):
    actions = q[state].keys()
    num_actions = len(actions)
    expl_prob = epsilon / num_actions
    greedy_action = choose_greedy_action(q, state)
    action_probs = [1 - epsilon + expl_prob if action == greedy_action else expl_prob for action in actions]
    return sum([action_prob * q[state][action] for (action, action_prob) in zip(actions, action_probs)])

def expected_sarsa(env, q, epsilon=0.1, alpha=0.5, gamma=1):
    for i in range(1000):
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
    goals = [(7,0)]
    anti_goals = [(1,0),(2,0),(3,0),(4,0),(5,0),(6,0)]
    env = Gridworld(8, 4, goals, anti_goals)

    # init q and get baseline random performance
    q = init_state_action_map(env)
    estimate_performance(env, q, 1)

    # learn q
    print("running expected sarsa...")
    q = expected_sarsa(env, q)
    print("expected sarsa complete")

    # determine post-training performance
    estimate_performance(env, q, 0.01)
    visualize_performance(env, q)

if __name__ == '__main__':
    main()
