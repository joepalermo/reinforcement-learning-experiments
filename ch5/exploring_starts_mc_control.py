import gym
import random
import numpy as np
import matplotlib
# setup matplotlib
matplotlib.use('TkAgg')
# import plot function
import matplotlib.pyplot as plt

# blackjack utilities ----------------------------------------------------------

# generate all states
def generate_states():
    for player_sum in xrange(4,22):
        for dealer_showing in xrange(1,11):
            for usable_ace in [False, True]:
                yield (player_sum, dealer_showing, usable_ace)

# generate all states with the supplied value of usable_ace
def generate_half_states(usable_ace=False):
    for player_sum in xrange(4,22):
        for dealer_showing in xrange(1,11):
            yield (player_sum, dealer_showing, usable_ace)

# generate all actions from a given state
def generate_actions():
    for action in xrange(0,2):
        yield action

# visualize the policy by way of the corresponding state-value function
def visualize_policy(q):
    states = [state for state in generate_half_states(usable_ace=False)]
    state_values = [q[state][1] for state in states]
    state_values_matrix = np.reshape(np.array(state_values), (18, 10))
    plt.imshow(state_values_matrix, cmap='hot', origin='lower', extent=((0,10,4,21)))
    plt.show()
    states = [state for state in generate_half_states(usable_ace=True)]
    state_values = [q[state][1] for state in states]
    state_values_matrix = np.reshape(np.array(state_values), (18, 10))
    plt.imshow(state_values_matrix, cmap='hot', origin='lower', extent=((0,10,4,21)))
    plt.show()

# generic utilities ------------------------------------------------------------

# initialize a map over all state-action pairs that maps each of them to 0
def init_state_action_map():
    state_action_map = dict()
    for state in generate_states():
        state_action_map[state] = dict()
        for action in generate_actions():
            state_action_map[state][action] = 0
    return state_action_map

# initialize a random deterministic policy
def init_policy():
    policy = dict()
    for state in generate_states():
        policy[state] = random.randint(0,1)
    return policy

# generate an episode represented as a list of tuples of form:
# (observation, action, reward, next_observation)
def generate_episode(env, policy):
    episode = list()
    observation = env.reset()
    #print "observation: " + str(observation)
    action = None
    while True:
        # at the start of an episode, select action ramdomly
        if action:
            action = choose_action(policy, observation)
        else:
            action = env.action_space.sample()
        #print "action: " + str(action)
        next_observation, reward, done, _ = env.step(action)
        episode_step = (observation, action, reward, next_observation)
        episode.append(episode_step)
        observation = next_observation
        if done:
            #print "reward: " + str(int(reward)) + "\n"
            observation = env.reset()
            break
    return episode

# choose an action using a deterministic policy
def choose_action(policy, observation):
    return policy[observation]

# perform episode-wise policy iteration
def policy_iteration(env, policy):
    q = init_state_action_map()
    visits_map = init_state_action_map()
    for _ in xrange(100000):
        episode = generate_episode(env, policy)
        policy_evaluation(episode, q, visits_map)
        policy_improvement(episode, q, policy)
    return q

# perform policy evaluation on an episode
def policy_evaluation(episode, q, visits_map):
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

# perform policy improvement on an episode
def policy_improvement(episode, q, policy):
    for (state, _, _, _) in episode:
        actions = [action for action in generate_actions()]
        best_action = (-1, -float("inf"))
        for i, action in enumerate(actions):
            value = q[state][action]
            if value > best_action[1]:
                best_action = (i, value)
        best_action = actions[best_action[0]]
        policy[state] = best_action

# main functionality -----------------------------------------------------------

def main():
    env = gym.make("Blackjack-v0")
    policy = init_policy()
    q = policy_iteration(env, policy)
    visualize_policy(q)

main()
