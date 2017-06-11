from Blackjack import Blackjack
from utilities import init_epsilon_greedy_policy

# test that state and action generation methods, generate the correct number
# of states and actions
def test_state_action_generation():
    env = Blackjack()
    assert len(list(env.generate_states())) == 360
    assert len(list(env.generate_half_states())) == 180
    assert len(list(env.generate_actions(None))) == 2

# test that the policy forms a conditional probability distribution on states
def test_init_epsilon_greedy_policy():
    env = Blackjack()
    epsilon = 0.1
    policy = init_epsilon_greedy_policy(env, epsilon)
    # for each state, the sum of the probability over the actions should sum
    # to a value of 1.0
    for state in env.generate_states():
        total_prob = 0
        for action in env.generate_actions(state):
            total_prob += policy[state][action]
        assert total_prob == 1.0


test_state_action_generation()
test_init_epsilon_greedy_policy()
