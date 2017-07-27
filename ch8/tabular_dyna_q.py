# boilerplate imports
import sys
from os.path import abspath, join, dirname
# add the top level package to sys.path to access utilities and environments
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
sys.path.insert(1, abspath(join(dirname(__file__), '../environments')))

# main imports
import time
from utilities import init_state_action_map, \
                      init_model, \
                      generate_random_episode, \
                      choose_greedy_action, \
                      choose_epsilon_greedy_action, \
                      max_q, \
                      generate_epsilon_greedy_episode
import random
from Maze import Maze

def tabular_dyna_q(env, q, n=5, epsilon=0.1, alpha=0.5, gamma=1, num_episodes=20):
    base_reward = 0
    model = init_model(env, base_reward)
    state_action_record = list()
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = choose_epsilon_greedy_action(q, state, epsilon)
            state_action_record.append((state, action))
            next_state, reward, done, _ = env.step(action)
            # update q
            td_error = reward + gamma * max_q(next_state, q) - q[state][action]
            q[state][action] += alpha * td_error
            # update model
            model[(state, action)] = (next_state, reward)
            state = next_state
            # planning phase
            for _ in range(n):
                (state_, action) = random.choice(state_action_record)
                (next_state, reward) = model[(state_, action)]
                print q[state_][action], reward
                # update q
                td_error = reward + gamma * max_q(next_state, q) - q[state_][action]
                q[state_][action] += alpha * td_error
    return q

def main():
    x_limit = 8
    y_limit = 5
    goals = [(0,4)]
    walls = [(0,2), (1,2), (2,2), (3,2)]

    env = Maze(x_limit, y_limit, goals, walls)
    num_episodes = 10

    # determine the baseline performance that results from taking random moves
    avg = sum([len(generate_random_episode(env)) for _ in range(num_episodes)]) / float(num_episodes)
    print "baseline random performance: " + str(avg)

    # learn q
    print "running tabular dyna-q..."
    q = init_state_action_map(env)
    q = tabular_dyna_q(env, q)
    print "tabular dyna-q complete"

    # evaluate performance
    avg = sum([len(generate_epsilon_greedy_episode(env, q)) for _ in range(num_episodes)]) / float(num_episodes)
    print "post learning performance: " + str(avg)

    # visualize post-training episode
    state = env.reset()
    while True:
        env.render()
        time.sleep(0.25)
        action = choose_epsilon_greedy_action(q, state, 0.1)
        state, _, done, _ = env.step(action) # take a random action
        if done:
            env.render(close=True)
            break

if __name__ == "__main__":
    main()
