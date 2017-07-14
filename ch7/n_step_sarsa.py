# boilerplate imports
import sys
from os.path import abspath, join, dirname
# add the top level package to sys.path to access utilities
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

# main imports
import time
from utilities import init_state_action_map, \
                      generate_random_episode, \
                      choose_greedy_action, \
                      choose_epsilon_greedy_action, \
                      generate_epsilon_greedy_episode
from Gridworld import Gridworld

def n_step_sarsa(env, alpha=0.5, epsilon=0.1, gamma=0.9, n=5, num_episodes=5000):
    q = init_state_action_map(env)
    for _ in range(num_episodes):
        # reset states, actions, and rewards lists
        states = []
        actions = []
        rewards = [None]
        # reset state, action
        state = env.reset()
        states.append(state)
        action = choose_epsilon_greedy_action(q, state, epsilon)
        actions.append(action)
        T = float("inf")
        t = 0
        while True:
            # while more actions remain to be taken
            if t < T:
                action = actions[t]
                next_state, reward, done, _ = env.step(action)
                states.append(next_state)
                rewards.append(reward)
                if done:
                    T = t+1
                else:
                    action = choose_epsilon_greedy_action(q, next_state, epsilon)
                    actions.append(action)
            # tau is the index on state/action updates
            tau = t - n + 1
            # if we are deep enough into an episode to perform an update
            if tau >= 0:
                # compute the target of the update (n-step return)
                G = sum([gamma ** (i-tau-1) * rewards[i] for i in range(tau+1, min(tau+n, T)+1)])
                if tau+n < T:
                    G = G + gamma ** n * q[states[tau+n]][actions[tau+n]]
                q_value = q[states[tau]][actions[tau]]
                # update the q function
                q[states[tau]][actions[tau]] = q_value + alpha * (G - q_value)
            # don't update the terminal state
            if tau == T - 1:
                break
            t = t + 1
    return q

def main():
    env = Gridworld(kings_moves=False)
    num_episodes = 1000

    # determine the baseline performance that results from taking random moves
    avg = sum([len(generate_random_episode(env)) for _ in range(num_episodes)]) / float(num_episodes)
    print "baseline random performance: " + str(avg)

    # learn q
    print "running n-step sarsa..."
    q = n_step_sarsa(env)
    print "n-step sarsa complete"

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
