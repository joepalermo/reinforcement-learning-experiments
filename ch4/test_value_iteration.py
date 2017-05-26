from value_iteration import *

# flag to indicate if you want to print test results that are most easily
# inspected visually
visual_test = False

if visual_test:
    for _ in range(25):
        print flip_coin()

# test action generator
assert len(list(generate_actions(0))) == 0
assert len(list(generate_actions(45))) == 45
assert len(list(generate_actions(55))) == 45
assert len(list(generate_actions(99))) == 1
assert len(list(generate_actions(100))) == 0
