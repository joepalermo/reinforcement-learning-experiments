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

# initialize a stochastic policy which selects all actions with equal probability
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

# action selection utilities ---------------------------------------------------

# choose an action using a deterministic policy
def choose_deterministic_action(policy, observation):
    return policy[observation]

# choose an action using a stochastic policy
def choose_stochastic_action(policy, observation):
    # action_dict is a map from actions to probability of selection
    action_dict = policy[observation]
    action_space = action_dict.keys()
    p_distn = action_dict.values()
    return np.random.choice(action_space, 1, p=p_distn)[0]

# select an epsilon greey action
def choose_epsilon_greedy_action(q, state, epsilon):
    actions = q[state].keys()
    if (random.random() < epsilon):
        return random.choice(actions)
    else:
        best_action_properties = (-1, -float("inf"))
        for i, action in enumerate(actions):
            if q[state][action] > best_action_properties[1]:
                best_action_properties = (i, q[state][action])
        best_action_i = best_action_properties[0]
        best_action = actions[best_action_i]
        return best_action

# select a greey action from a given state
def choose_greedy_action(q, state):
    actions = q[state].keys()
    best_action_properties = (-1, -float("inf"))
    for i, action in enumerate(actions):
        if q[state][action] > best_action_properties[1]:
            best_action_properties = (i, q[state][action])
    best_action_i = best_action_properties[0]
    best_action = actions[best_action_i]
    return best_action

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

# generate a random episode, and return a list of its states
def generate_random_episode(env):
    state = env.reset()
    episode = [state]
    done = False
    while not done:
        random_action = env.action_space.sample()
        (next_state, reward, done, _) = env.step(random_action)
        episode.append(next_state)
    return episode

def generate_greedy_episode(env, q):
    state = env.reset()
    episode = [state]
    done = False
    while not done:
        greedy_action = choose_greedy_action(q, state)
        (next_state, reward, done, _) = env.step(greedy_action)
        episode.append(next_state)
        state = next_state
    return episode

def generate_epsilon_greedy_episode(env, q, epsilon=0.1):
    state = env.reset()
    episode = [state]
    done = False
    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = choose_greedy_action(q, state)
        (next_state, reward, done, _) = env.step(action)
        episode.append(next_state)
        state = next_state
    return episode

# policy evaluation utilities --------------------------------------------------

# get the importance sampling ratio for a given state action pair
def importance_sampling(target_policy, behavior_policy, state, action):
    return target_policy[state][action] / behavior_policy[state][action]

def on_policy_state_evaluation(episode, v, visits_map):
    n = len(episode)
    ret = 0
    updates_map = dict()
    for i in range(n-1, -1, -1):
        (state, _, reward, _) = episode[i]
        ret += reward
        m = visits_map[state] + 1
        updated_value = v[state] + 1.0 / m * (ret - v[state])
        updates_map[state] = updated_value
    for state in updates_map:
        visits_map[state] += 1
        v[state] = updates_map[state]

# perform episode-wise first-visit on-policy evaluation
def on_policy_evaluation(episode, q, visits_map):
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

# perform episode-wise off-policy every-visit policy evaluation
def off_policy_evaluation(episode, q, c, target_policy, behavior_policy, gamma=1):
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

# policy improvement utilities -------------------------------------------------

# perform episode-wise epsilon-greedy policy improvement
def epsilon_greedy_policy_improvement(env, episode, q, policy, epsilon):
    for (state, _, _, _) in episode:
        actions = [action for action in env.generate_actions(state)]
        num_actions = len(actions)
        exploratory_action_prob = epsilon / num_actions
        best_action = (-1, -float("inf"))
        for i, action in enumerate(actions):
            # default all action probabilities to the baseline
            policy[state][action] = exploratory_action_prob
            action_value = q[state][action]
            if action_value > best_action[1]:
                best_action = (i, action_value)
        best_action_i = best_action[0]
        best_action = actions[best_action_i]
        policy[state][best_action] = 1 - epsilon + exploratory_action_prob
        assert policy[state][best_action] + (num_actions - 1) * exploratory_action_prob == 1.0

# perform episode-wise policy improvement for a deterministic policy
def greedy_deterministic_policy_improvement(env, episode, q, policy):
    for (state, _, _, _) in episode:
        actions = [action for action in env.generate_actions(state)]
        best_action = (-1, -float("inf"))
        for i, action in enumerate(actions):
            value = q[state][action]
            if value > best_action[1]:
                best_action = (i, value)
        best_action = actions[best_action[0]]
        policy[state] = best_action

# perform episode-wise greedy policy improvement on a policy represented stochastically
def greedy_stochastic_policy_improvement(env, episode, q, policy):
    for (state, _, _, _) in episode:
        actions = [action for action in env.generate_actions(state)]
        best_action = (-1, -float("inf"))
        for i, action in enumerate(actions):
            policy[state][action] = 0
            value = q[state][action]
            if value > best_action[1]:
                best_action = (i, value)
        best_action = actions[best_action[0]]
        policy[state][best_action] = 1.0

# policy iteration utilities ---------------------------------------------------

# perform episode-wise off-policy every-visit, fine-grained policy iteration
# interleave policy evaluation and policy improvement on a per-update basis
def fine_grained_off_policy_iteration(episode, q, c, target_policy, behavior_policy, gamma=1):
    g = 0
    w = 1.0
    num_steps = len(episode)
    for i in xrange(num_steps-1, -1, -1):
        (state, action, reward, next_state) = episode[i]
        g = gamma * g + reward
        c[state][action] += w
        q[state][action] += w / c[state][action] * (g - q[state][action])
        # improve policy based on q update
        actions = q[state].keys()
        best_action_value = (-1, -float("inf"))
        for i, a in enumerate(actions):
            # by default set each action to a non-greedy state
            target_policy[state][a] = 0
            action_value = (i, q[state][a])
            if action_value[1] > best_action_value[1]:
                best_action_value = action_value
        best_action_i = best_action_value[0]
        best_action = actions[best_action_i]
        q[state][best_action] = 1.0
        # if the current action wouldn't be taken now, break
        if best_action != action:
            break
        # if the current action is taken now, then its probability under
        # the target policy is 1
        w *= 1.0 / behavior_policy[state][action]
