import numpy as np

'''generic utility methods for implementing RL algorithms'''

# initialize a map over all states that maps every state to 0
def init_state_map(env):
    state_map = dict()
    for state in env.generate_states():
        state_map[state] = 0
    return state_map

# initialize a map over all state-action pairs that maps each of them to 0
def init_state_action_map(env):
    state_action_map = dict()
    for state in env.generate_states():
        state_action_map[state] = dict()
        for action in env.generate_actions():
            state_action_map[state][action] = 0
    return state_action_map

# initialize a random deterministic policy
def init_deterministic_policy(env):
    policy = dict()
    for state in env.generate_states():
        policy[state] = env.action_space.sample()
    return policy

# choose an action using a deterministic policy
def choose_deterministic_action(policy, observation):
    return policy[observation]

# choose an action
def choose_stochastic_action(policy, observation):
    # action_dict is a map from actions to probability of selection
    action_dict = policy[observation]
    action_space = action_dict.keys()
    p_distn = action_dict.values()
    return np.random.choice(action_space, 1, p=p_distn)[0]
