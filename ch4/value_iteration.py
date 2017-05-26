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

# generic utilities ------------------------------------------------------------

# compare floats for approximate equality
def is_close(a, b, abs_tol=1e-10):
    return abs(a-b) <= abs_tol

# core functionality -------------------------------------------------------------------

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

def value_function_init():
    v = dict()
    for state in generate_states():
        v[state] = 0
    return v

def state_update(v, state, gamma=0.9):
    max_value = 0
    # find the max value over actions from a given state
    for action in generate_actions(state):
        value = 0
        for (next_state, reward, p_outcome) in generate_outcomes(state, action):
            value += p_oucome * (reward + gamma * v[next_state])
        if value > max_value:
            max_value = value
    return max_value

def construct_optimal_policy(vf):
    pass

def value_iteration(theta = 0.5):
    v = value_function_init()
    # perform value iteration
    while True:
        max_delta = 0
        for state in generate_states():
            v = v[state]
            v[state] = state_update(state)
            delta = abs(v - v[state])
            if delta > max_delta:
                max_delta = delta
        if max_delta < theta:
            break
    optimal_policy = construct_optimal_policy(vf)
    return optimal_policy
    
#value_iteration()
