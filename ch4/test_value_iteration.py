from value_iteration import *

# assume that max_state is 100

# flag to indicate if you want to print test results that are most easily
# inspected visually
visual_test = False

if visual_test:
    for _ in range(25):
        print flip_coin()

# test state generator
assert len(list(generate_states())) == 101

# test action generator
assert len(list(generate_actions(0))) == 0
assert len(list(generate_actions(45))) == 45
assert len(list(generate_actions(55))) == 45
assert len(list(generate_actions(99))) == 1
assert len(list(generate_actions(100))) == 0

# test outcome generator
for state in generate_states():
    for action in generate_actions(state):
        total_prob = 0
        for (next_state, reward, probability) in generate_outcomes(state, action):
            total_prob += probability
        assert total_prob == 1.0

# test value function initialization
v = value_function_init()
assert len(v) == max_state + 1
