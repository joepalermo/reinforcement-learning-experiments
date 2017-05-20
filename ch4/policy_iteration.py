# some of this code is copied and/or inspired by https://github.com/kkleidal
# https://github.com/kkleidal/SuttonRLExercises/blob/master/chapters/4/ex4_5.py

debug = True
max_cars = 10

from math import factorial, e

def poisson(expected, actual):
    return (float(expected) ** actual / factorial(actual)) * e ** (-expected)

def poisson_truncated(expected, actual, maximum):
    if actual >= maximum:
        return 1 - sum([poisson(expected, i) for i in xrange(0,maximum)])
    else:
        return poisson(expected, actual)

def generate_states():
    for cars_1 in xrange(1,max_cars+1):
        for cars_2 in xrange(1,max_cars+1):
            yield (cars_1, cars_2)

def generate_actions(state):
    (cars_1, cars_2) = state
    # no more than 5 cars can be moved across locations overnight
    min_action = min(5, cars_2)
    max_action = min(5, cars_1)
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

def generate_outcomes(state, action, expected_rentals, expected_returns):
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
            reward = - 2 * abs(action) + (n_rent_1 + n_rent_2) * 10
            # return the outcome, and its probability
            outcome = (s_prime, reward)
            p_outcome = p_rent * p_return
            yield outcome, p_outcome


# Construct the conditional probability distribution, p(s',r|s,a).
# Represent it as a dictionary mapping conditions (s,a) to dictionaries
# each of which maps (next state, reward) pairs to a probability
def construct_p(expected_rentals=(3,4), expected_returns=(3,2)):
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
            for outcome, p_outcome in generate_outcomes(state, action, expected_rentals, expected_returns):
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
        min_action = - min(5, cars_2)
        max_action = min(5, cars_1)
        # for all valid actions from a given state
        for a in generate_actions(state):
            policy[state][a] = 1.0 / max_action+1 - (min_action)

def initialize_policy_value():
    policy_value = dict()
    for state in generate_states():
        policy_value[state] = 0
    return policy_value

def policy_evaluation(p, policy, theta = 0.01):
    policy_value = initialize_policy_value()
    while True:
        max_delta = 0
        for state in generate_states():
            v = policy_value(state)
            v_update = state_update(p, policy_value, state)
            delta = abs(v - v_update)
            if delta > max_delta:
                max_delta = delta
        if max_delta < theta:
            break
    return policy_value

def state_update(p, policy_value, state, gamma = 0.9):
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


def policy_improvement(policy, policy_value):
    pass
    # todo
    return policy, policy_stable

def main():
    p = construct_p()

    # represent a policy as a map from state to a map from action to probability
    # policy = initialize_policy()
    # policy_stable = False
    # while not policy_stable:
    #     policy_value = policy_evaluation(p, policy)
    #     policy, policy_stable = policy_improvement(policy, policy_value)
    #
    # print 'Behold, the optimal policy...'
    # print policy


main()
