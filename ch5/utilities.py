import random
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
        for action in env.generate_actions(state):
            state_action_map[state][action] = 0
    return state_action_map

# initialize a random deterministic policy
def init_deterministic_policy(env):
    policy = dict()
    for state in env.generate_states():
        policy[state] = env.action_space.sample()
    return policy

# return a tuple containing a mid-episode environment, the environment's
# current state, and a randomly selected action from that state
def get_random_state_action(env):
    (env, state) = get_mid_episode_state(env)
    actions = [action for action in env.generate_actions(state)]
    action = random.choice(actions)
    return (env, state, action)

# return a (state, environment) pair mid-episode
def get_mid_episode_state(env):
    observation = env.reset()
    done = False
    while True:
        if random.random() < 0.5:
            return (env, observation)
        action = env.action_space.sample()
        next_observation, reward, done, _ = env.step(action)
        observation = next_observation
        if done:
            observation = env.reset()

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
