# some of this code is copied and/or inspired by https://github.com/kkleidal
# https://github.com/kkleidal/SuttonRLExercises/blob/master/chapters/4/ex4_5.py

import numpy as np
import matplotlib
# setup matplotlib
matplotlib.use('TkAgg')
# import plot function
import matplotlib.pyplot as plt
from math import factorial, e

# global variables -------------------------------------------------------------

debug = False
max_cars = 5
max_car_movement = 5
move_penalty = 2
rental_reward = 10
expected_rentals=(3,4)
expected_returns=(3,2)

# generic utilities ------------------------------------------------------------

# compare floats for approximate equality
def isclose(a, b, abs_tol=1e-10):
    return abs(a-b) <= abs_tol

def poisson(expected, actual):
    return (float(expected) ** actual / factorial(actual)) * e ** (-expected)

def poisson_truncated(expected, actual, maximum):
    if actual >= maximum:
        return 1 - sum([poisson(expected, i) for i in xrange(0,maximum)])
    else:
        return poisson(expected, actual)

# generators -------------------------------------------------------------------

# a generator that produces all possible states
def generate_states():
    for cars_1 in xrange(0,max_cars+1):
        for cars_2 in xrange(0,max_cars+1):
            yield (cars_1, cars_2)

# a generator that given a state produces all possible legal actions
def generate_actions(state):
    (cars_1, cars_2) = state
    # no more than 5 cars can be moved across locations overnight
    min_action = min(max_car_movement, cars_2)
    max_action = min(max_car_movement, cars_1)
    for from_2_to_1 in range(1, min_action+1):
        if from_2_to_1 + cars_1 > max_cars:
            break
        yield -from_2_to_1
    for from_1_to_2 in range(1, max_action+1):
        if from_1_to_2 + cars_2 > max_cars:
            break
        yield from_1_to_2
    yield 0

# a generator that given a sequence type of length 2, will produce all pairs of
# values that can be produced by combining values in the range from 0 to the
# the values in the sequence type
def generate_paired_outcomes(limits):
    for n_1 in xrange(0, limits[0]+1):
        for n_2 in xrange(0, limits[1]+1):
            yield (n_1, n_2)

# a generator that given a state and action will enumerate all possibilities for
# the environment's response and return (outcome, probability) pairs. Note that
# the value probability is not the probability of the outcome, since some
# outcomes can be generated in more than one way
def generate_possibilities(state, action):
    (rent_1, rent_2) = expected_rentals
    (return_1, return_2) = expected_returns
    (cars_1, cars_2) = state
    cars_to_rent = (cars_1 - action, cars_2 + action)
    # enumerate the possibilities for rentals
    for (n_rent_1, n_rent_2) in generate_paired_outcomes(cars_to_rent):
        p_n_rent_1 = poisson_truncated(rent_1, n_rent_1, cars_to_rent[0])
        p_n_rent_2 = poisson_truncated(rent_2, n_rent_2, cars_to_rent[1])
        p_rent = p_n_rent_1 * p_n_rent_2
        cars_remaining = (cars_to_rent[0] - n_rent_1,
                          cars_to_rent[1] - n_rent_2)
        # enumerate the possibilities for returns
        max_returns = (max_cars - cars_remaining[0], max_cars - cars_remaining[1])
        for (n_return_1, n_return_2) in generate_paired_outcomes(max_returns):
            p_n_return_1 = poisson_truncated(return_1, n_return_1, max_returns[0])
            p_n_return_2 = poisson_truncated(return_2, n_return_2, max_returns[1])
            p_return = p_n_return_1 * p_n_return_2
            # compute s' and r
            s_prime = (cars_remaining[0] + n_return_1, cars_remaining[1] + n_return_2)
            reward = - move_penalty * abs(action) + (n_rent_1 + n_rent_2) * rental_reward
            # return the outcome, and the probability of this contingency
            outcome = (s_prime, reward)
            probability = p_rent * p_return
            if debug:
                print
                print "condition: " + str((state, action))
                print "rentals: " + str((n_rent_1, n_rent_2))
                print "returns: " + str((n_return_1, n_return_2))
                print "outcome: " + str(outcome) + ", probability: " + str(probability)
                print
            yield outcome, probability

# core functionality -----------------------------------------------------------

# Construct the conditional probability distribution, p(s',r|s,a).
# Represent it as a dictionary mapping conditions (s,a) to dictionaries
# each of which maps (next state, reward) pairs to a probability
def construct_p():
    p = dict()
    # for all states
    for state in generate_states():
        # for all valid actions from a given state
        for action in generate_actions(state):
            # construct the condition
            condition = (state, action)
            # compute the conditional probability distribution for a given condition
            p_cond = dict()
            p[condition] = p_cond
            # accumulate total probability over outcomes to ensure that p
            # forms a valid probability distribution
            total_prob = 0
            for outcome, probability in generate_possibilities(state, action):
                p_cond[outcome] = 0
            for outcome, probability in generate_possibilities(state, action):
                total_prob += probability
                # assign probability to the conditional outcome
                p_cond[outcome] += probability
            if not isclose(total_prob, 1.0):
                raise Exception("p doesn't form a probability distribution")
    print "done constructing the environment's probability distribution"
    return p

# return an equiprobable policy
def initialize_policy():
    policy = dict()
    for state in generate_states():
        policy[state] = dict()
        (cars_1, cars_2) = state
        # no more than 5 cars can be moved across locations overnight
        min_action = - min(max_car_movement, cars_2)
        max_action = min(max_car_movement, cars_1)
        # for all valid actions from a given state
        actions = [a for a in generate_actions(state)]
        for a in actions:
            policy[state][a] = 1.0 / len(actions)
    return policy

# return a value function that is 0 for all states
def initialize_policy_value():
    policy_value = dict()
    for state in generate_states():
        policy_value[state] = 0
    return policy_value

# takes as input a value function (represented as a dictionary mapping states
# to expected value) and displays a visualization of the value function as a
# heatmap
def visualize_value_function(policy_value):
    states = [state for state in generate_states()]
    state_values = [policy_value[state] for state in states]
    state_values_arr = np.array(state_values)
    state_values_matrix = np.reshape(state_values_arr, (max_cars+1, max_cars+1))
    print "min state value: " + str(state_values_matrix.min()) + \
          ", max state value: " + str(state_values_matrix.max())
    plt.imshow(state_values_matrix, cmap='hot', origin='lower')
    plt.show()

# takes as input a policy, performs policy evaluation and returns the value
# function for the policy
def policy_evaluation(policy, theta = 0.5):
    policy_value = initialize_policy_value()
    while True:
        max_delta = 0
        for state in generate_states():
            v = policy_value[state]
            v_update = state_update(policy, policy_value, state)
            delta = abs(v - v_update)
            if delta > max_delta:
                max_delta = delta
        if debug:
            print "max delta: " + str(max_delta)
        if max_delta < theta:
            break
    return policy_value

# given a policy, a value function for the policy, and a particular state,
# update the value function for that state. Also return the value of the update
def state_update(policy, policy_value, state, gamma = 0.9):
    (cars_1, cars_2) = state
    value_sum = 0
    for action in generate_actions(state):
        p_cond = p[(state, action)]
        for outcome in p_cond:
            next_state = outcome[0]
            reward = outcome[1]
            value_sum += policy[state][action] * p_cond[outcome] * (reward + gamma * policy_value[next_state])
    policy_value[state] = value_sum
    return value_sum

# given a policy and its value function return an improved policy, and an
# indication of policy stability
def policy_improvement(policy, policy_value, gamma = 0.9):
    new_policy = dict()
    policy_stable = True
    for state in generate_states():
        new_policy[state] = dict()
        # represent best action as a tuple with its index and action value
        best_action = (-1, 0)
        actions = [action for action in generate_actions(state)]
        for action_i, action in enumerate(actions):
            condition = (state, action)
            p_cond = p[condition]
            # by default, in the new policy let the probability of a given
            # action in a given state be 0
            new_policy[state][action] = 0
            # compute an estimate of the value of the action under consideration
            action_value = 0
            for outcome in p_cond:
                next_state = outcome[0]
                reward = outcome[1]
                action_value += p_cond[outcome] * (reward + gamma * policy_value[next_state])
            # if this action is the best action, take note of that
            best_action_value = best_action[1]
            if action_value > best_action_value:
                best_action = (action_i, action_value)
        # give the best action probability 1 in the new policy
        best_action_index = best_action[0]
        best_action = actions[best_action_index]
        new_policy[state][best_action] = 1.0
        # check whether the new policy is different from the prior policy, and
        # update policy stability accordingly
        if policy[state][best_action] != 1.0:
            policy_stable = False
    return new_policy, policy_stable

def main():
    # represent a policy as a map from state to a map from action to probability
    policy = initialize_policy()
    policy_stable = False
    while not policy_stable:
        policy_value = policy_evaluation(policy)
        policy, policy_stable = policy_improvement(policy, policy_value)
    visualize_value_function(policy_value)
    print 'Behold, the optimal policy... ' + str(policy)

# construct the environment's probability distribution
p = construct_p()
# find the optimal policy
main()
