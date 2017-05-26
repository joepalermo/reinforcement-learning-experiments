import random
import numpy as np
import matplotlib
# setup matplotlib
matplotlib.use('TkAgg')
# import plot function
import matplotlib.pyplot as plt

# global variables -------------------------------------------------------------

max_state = 100
p_heads = 0.25
gamma=0.9

# generators -------------------------------------------------------------------

def generate_states():
    for s in xrange(0, max_state+1):
        yield s

def generate_actions(state):
    max_action = min(state, max_state - state)
    for a in xrange(1, max_action+1):
        yield a

def generate_outcomes(state, action):
    for result in ['heads', 'tails']:
        if result == 'heads':
            next_state = state + action
            if next_state == max_state:
                reward = 1
            else:
                reward = 0
            probability = p_heads
        else:
            next_state = state - action
            reward = 0
            probability = 1 - p_heads
        yield (next_state, reward, probability)

# core functionality -----------------------------------------------------------

def value_function_init():
    v = dict()
    for state in generate_states():
        v[state] = 0
    return v

def state_update(v, state):
    max_value = 0
    # find the max value over actions from a given state
    for action in generate_actions(state):
        value = 0
        for (next_state, reward, p_outcome) in generate_outcomes(state, action):
            value += p_outcome * (reward + gamma * v[next_state])
        if value > max_value:
            max_value = value
    return max_value

def construct_optimal_policy(v):
    policy = dict()
    for state in generate_states():
        policy[state] = dict()
        actions = list(generate_actions(state))
        # keep track of the current best action from a given state
        max_action_i = -1
        max_action_value = 0
        for action_i, action in enumerate(actions):
            # by default let the policy map actions to zero probability
            policy[state][action] = 0
            action_value = 0
            for (next_state, reward, p_outcome) in generate_outcomes(state, action):
                action_value += p_outcome * (reward + gamma * v[next_state])
            if action_value > max_action_value:
                max_action_i = action_i
                max_action_value = action_value
        if actions:
            max_action = actions[max_action_i]
            policy[state][max_action] = 1
    return policy

def value_iteration(theta = 0.01):
    v = value_function_init()
    # perform value iteration
    i = 0
    while True:
        print "value iteration # " + str(i)
        max_delta = 0
        for state in generate_states():
            state_value = v[state]
            v[state] = state_update(v, state)
            delta = abs(state_value - v[state])
            if delta > max_delta:
                max_delta = delta
        print "max delta: " + str(max_delta)
        if max_delta < theta:
            break
        i += 1
    optimal_policy = construct_optimal_policy(v)
    return optimal_policy, v

def visualize_value_function(v):
    data_to_plot = [v[state] for state in generate_states()]
    plt.plot(data_to_plot)
    plt.show()

def main():
    opt_policy, v = value_iteration()
    visualize_value_function(v)

    
main()
