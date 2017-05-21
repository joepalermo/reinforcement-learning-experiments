# some of this code is copied and/or inspired by https://github.com/kkleidal
# https://github.com/kkleidal/SuttonRLExercises/blob/master/chapters/4/ex4_5.py

debug = False
max_cars = 5
max_cars_move = 1
move_penalty = 5
expected_rentals=(1,1)
rental_reward = 10
expected_returns=(1,1)

import numpy as np
import matplotlib
# setup matplotlib
matplotlib.use('TkAgg')
# import plot function
import matplotlib.pyplot as plt
from math import factorial, e

def poisson(expected, actual):
    return (float(expected) ** actual / factorial(actual)) * e ** (-expected)

def poisson_truncated(expected, actual, maximum):
    if actual >= maximum:
        return 1 - sum([poisson(expected, i) for i in xrange(0,maximum)])
    else:
        return poisson(expected, actual)

def visualize_value_function(policy_value):
    states = [state for state in generate_states()]
    state_values = [policy_value[state] for state in states]
    state_values_arr = np.array(state_values)
    state_values_matrix = np.reshape(state_values_arr, (max_cars+1, max_cars+1))
    # debug
    print "min: " + str(state_values_matrix.min()) + ", max: " + str(state_values_matrix.max())
    #plt.imshow(state_values_matrix, cmap='hot', origin='lower')
    #plt.show()

def generate_states():
    for cars_1 in xrange(0,max_cars+1):
        for cars_2 in xrange(0,max_cars+1):
            yield (cars_1, cars_2)

def generate_actions(state):
    (cars_1, cars_2) = state
    # no more than 5 cars can be moved across locations overnight
    min_action = min(max_cars_move, cars_2)
    max_action = min(max_cars_move, cars_1)
    for from_2_to_1 in range(1, min_action+1):
        if from_2_to_1 + cars_1 > max_cars:
            break
        yield -from_2_to_1
    for from_1_to_2 in range(1, max_action+1):
        if from_1_to_2 + cars_2 > max_cars:
            break
        yield from_1_to_2
    yield 0

def generate_paired_outcomes(limits):
    for n_1 in xrange(0, limits[0]+1):
        for n_2 in xrange(0, limits[1]+1):
            yield (n_1, n_2)

def generate_outcomes(state, action):
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
            # print statements for debugging
            if debug:
                print "condition: " + str((state, action))
                print "rentals: " + str((n_rent_1, n_rent_2))
                print "returns: " + str((n_return_1, n_return_2))
            # compute s' and r
            s_prime = (cars_remaining[0] + n_return_1, cars_remaining[1] + n_return_2)
            reward = - move_penalty * abs(action) + (n_rent_1 + n_rent_2) * rental_reward
            # return the outcome, and its probability
            outcome = (s_prime, reward)
            p_outcome = p_rent * p_return
            yield outcome, p_outcome


# Construct the conditional probability distribution, p(s',r|s,a).
# Represent it as a dictionary mapping conditions (s,a) to dictionaries
# each of which maps (next state, reward) pairs to a probability
def construct_p():
    p = dict()
    # for all states
    i = 0
    for state in generate_states():
        # counter to measure progress towards completion
        i += 1
        print i
        # for all valid actions from a given state
        for action in generate_actions(state):
            # construct the condition
            condition = (state, action)
            # compute the conditional probability distribution for a given condition
            p_cond = dict()
            p[condition] = p_cond
            for outcome, p_outcome in generate_outcomes(state, action):
                    # print statements for debugging
                    if debug:
                        print "outcome: " + str(outcome) + ", p_outcome: " + str(p_outcome)
                        print
                    # assign probability to the conditional outcome
                    p_cond[outcome] = p_outcome

    return p

# return an equiprobable policy
def initialize_policy():
    policy = dict()
    for state in generate_states():
        policy[state] = dict()
        (cars_1, cars_2) = state
        # no more than 5 cars can be moved across locations overnight
        min_action = - min(max_cars_move, cars_2)
        max_action = min(max_cars_move, cars_1)
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

# given a model of the environment and a policy, perform policy evaluation
def policy_evaluation(p, policy, theta = 5):
    policy_value = initialize_policy_value()
    i = 0
    while True:
        #visualize_value_function(policy_value)
        # debug
        #print "pol_eval: " + str(i)
        i += 1
        max_delta = 0
        for state in generate_states():
            v = policy_value[state]
            v_update = state_update(p, policy, policy_value, state)
            # debug
            # if state == (5,5):
            #     print v, v_update
            delta = abs(v - v_update)
            if delta > max_delta:
                max_delta = delta
        # debug
        print "max_delta: " + str(max_delta)

        if max_delta < theta:
            break
    return policy_value

# given a model of the environment, a value function for a given policy, and
# some state, update the policy value function for that state
def state_update(p, policy, policy_value, state, gamma = 0.9):
    (cars_1, cars_2) = state
    value_sum = 0
    for action in generate_actions(state):
        p_cond = p[(state, action)]
        for (outcome, _) in generate_outcomes(state, action):
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
        # debug
        #print "state: " + str(state)
        new_policy[state] = dict()
        # represent best action as a tuple with its index and action value
        best_action = (-1, 0)
        actions = [action for action in generate_actions(state)]
        for action_i, action in enumerate(actions):
            # by default, in the new policy let the probability of a given
            # action in a given state be 0
            new_policy[state][action] = 0
            # compute an estimate of the value of the action under consideration
            action_value = 0
            for outcome, p_outcome in generate_outcomes(state, action):
                next_state = outcome[0]
                reward = outcome[1]
                action_value += p_outcome * (reward + gamma * policy_value[next_state])
            # debug
            #print "action: " + str(action) + ", action_value: " + str(action_value)
            # if this action is the best action, take note of that
            best_action_value = best_action[1]
            if action_value > best_action_value:
                best_action = (action_i, action_value)
        # give the best action probability 1 in the new policy
        best_action_i = best_action[0]
        best_action = actions[best_action_i]
        new_policy[state][best_action] = 1.0
        # debug
        # for a in actions:
        #     print "action: " + str(a) + ", prob: " + str(new_policy[state][a])
        # check whether the new policy is different from the prior policy
        for action in generate_actions(state):
            if policy[state][action] != new_policy[state][action]:
                policy_stable = False
    return new_policy, policy_stable

def main():
    p = construct_p()

    # represent a policy as a map from state to a map from action to probability
    policy = initialize_policy()
    policy_stable = False
    while not policy_stable:
        policy_value = policy_evaluation(p, policy)
        # debug
        # min_state_value = float("inf")
        # for state in generate_states():
        #     state_value = policy_value[state]
        #     if state_value < min_state_value:
        #         min_state_value = state_value
        # print "min_value " + str(min_state_value)

        policy, policy_stable = policy_improvement(policy, policy_value)

    print 'Behold, the optimal policy...'
    print policy


main()
