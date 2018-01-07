import sys
import random
from os.path import abspath, join, dirname
# add the top level package to sys.path to access utilities and environments
sys.path.insert(1, abspath(join(dirname(__file__), '../environments')))
from TicTacToe import TicTacToe, init_value_function, get_random_action, \
                      choose_greedy_action, get_immutable_state, print_state

# main functionality -----------------------------------------------------------

# epsilon: exploration parameter, alpha: step-size
def learn_tic_tac_toe(env, epsilon, alpha, num_episodes):
    value_function = init_value_function()
    for i in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            if random.random() < epsilon:
                action = get_random_action(state)
                update = False
            else:
                action = choose_greedy_action(value_function, state)
                update = True
            # note that reward is not needed due to value function initialization
            (next_state, _, done, _) = env.step(action)
            if update:
                value_function[get_immutable_state(state)] += alpha * \
                (value_function[get_immutable_state(next_state)] - value_function[get_immutable_state(state)])
            state = next_state
    return value_function

def evaluate_performance(value_function, env, num_episodes):
    record = [0,0,0]
    for i in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = choose_greedy_action(value_function, state)
            (next_state, reward, done, _) = env.step(action)
            state = next_state
        if reward == 1:
            record[0] += 1
        elif reward == 0:
            record[1] += 1
        else:
            record[2] += 1
    return record

epsilon = 0.2
alpha = 0.05
# create a new TicTacToe environment/opponent
env = TicTacToe()
# evaluate performance without any learning
value_function = learn_tic_tac_toe(env, epsilon, alpha, 0)
wins, ties, losses = evaluate_performance(value_function, env, 1000)
print(wins, ties, losses)
# train for 10k episodes
num_episodes = 20000
value_function = learn_tic_tac_toe(env, epsilon, alpha, num_episodes)
# evaluate performance post learning
wins, ties, losses = evaluate_performance(value_function, env, 1000)
print(wins, ties, losses)
