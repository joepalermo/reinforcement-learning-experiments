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

def flip_coin():
    if random.random() <= p_heads:
        return "heads"
    else:
        return "tails"

def generate_actions(state):
    max_action = min(state, max_state - state)
    for a in xrange(1, max_action+1):
        yield a

        

# core functionality -------------------------------------------------------------------

def value_function_init():
    pass

def state_update():
    pass

def construct_optimal_policy(vf):
    pass


def value_iteration(theta = 0.5):
    v = value_function_init()
    # perform value iteration
    while True:
        max_delta = 0
        for state in xrange(0, max_state + 1):
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
