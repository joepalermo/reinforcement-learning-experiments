import random
import numpy as np

'''generic utility methods for implementing RL algorithms'''

# initialization utilities -----------------------------------------------------

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

def init_equiprobable_random_policy(env):
    policy = dict()
    for state in env.generate_states():
        policy[state] = dict()
        actions = [action for action in env.generate_actions(state)]
        num_actions = len(actions)
        for action in actions:
            policy[state][action] = 1.0 / num_actions
    return policy

# initialize an epsilon-greedy policy
# epsilon must be in the domain [0, 1)
def init_epsilon_greedy_policy(env, epsilon):
    policy = dict()
    for state in env.generate_states():
        policy[state] = dict()
        actions = [action for action in env.generate_actions(state)]
        num_actions = len(actions)
        # assign all actions to have at least a baseline probability
        exploratory_action_prob = epsilon / num_actions
        for action in actions:
            policy[state][action] = exploratory_action_prob
        # randomly select an action to be initialized as the greedy action
        greedy_action_i = random.randint(0, num_actions-1)
        greedy_action = actions[greedy_action_i]
        policy[state][greedy_action] = 1 - epsilon + exploratory_action_prob
    return policy

# episode generation utilities -------------------------------------------------

# return a tuple containing a mid-episode environment, the environment's
# current state, and a randomly selected action from that state
def get_random_state_action(env):
    (env, state) = get_mid_episode_state(env)
    actions = [action for action in env.generate_actions(state)]
    action = random.choice(actions)
    return (env, state, action)

# return a (environment, state) pair mid-episode
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

# generate an episode represented as a list of tuples of form:
# (observation, action, reward, next_observation)
def generate_episode(env, policy):
    episode = list()
    observation = env.reset()
    done = False
    while not done:
        action = choose_stochastic_action(policy, observation)
        next_observation, reward, done, _ = env.step(action)
        episode_step = (observation, action, reward, next_observation)
        episode.append(episode_step)
        observation = next_observation
    return episode

# generate an episode using exploring starts
# return the episode represented as a list of tuples of form:
# (observation, action, reward, next_observation)
def generate_episode_es(env, policy):
    episode = list()
    (env, observation, action) = get_random_state_action(env)
    done = False
    while not done:
        if action == None:
            action = choose_deterministic_action(policy, observation)
        next_observation, reward, done, _ = env.step(action)
        episode_step = (observation, action, reward, next_observation)
        episode.append(episode_step)
        observation = next_observation
        action = None
    return episode

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

# policy evaluation/improvement/iteration utilities ----------------------------

def importance_sampling(target_policy, behavior_policy, state, action):
    return target_policy[state][action] / behavior_policy[state][action]

# perform on-policy evaluation on an episode
def on_policy_episode_eval(episode, q, visits_map):
    n = len(episode)
    ret = 0
    updates_map = dict()
    for i in range(n-1, -1, -1):
        (state, action, reward, _) = episode[i]
        ret += reward
        m = visits_map[state][action] + 1
        updated_value = q[state][action] + 1.0 / m * (ret - q[state][action])
        updates_map[(state, action)] = updated_value
    for (state, action) in updates_map:
        visits_map[state][action] += 1
        q[state][action] = updates_map[(state, action)]

# perform off-policy evaluation on an episode
def off_policy_episode_evaluation(episode, q, c, target_policy, behavior_policy):
    g = 0
    w = 1.0
    num_steps = len(episode)
    for i in xrange(num_steps-1, -1, -1):
        (state, action, reward, next_state) = episode[i]
        g = gamma * g + reward
        c[state][action] += w
        q[state][action] += w / c[state][action] * (g - q[state][action])
        w *= importance_sampling(target_policy, behavior_policy, state, action)
        if w == 0:
            break
