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

# initialize a policy that only sticks on 20 or 21
def init_policy():
    policy = dict()
    for state in generate_states():
        policy[state] = dict()
        player_sum = state[0]
        if player_sum < 20:
            policy[state][1] = 1.0
            policy[state][0] = 0
        else:
            policy[state][1] = 0
            policy[state][0] = 1.0
    return policy

# takes as input a value function (represented as a dictionary mapping states
# to expected value) and displays a visualization of the value function as a
# heatmap
def visualize_value_function(policy_value):
    # visualize states where usable_ace is False
    states = [state for state in generate_half_states(usable_ace=False)]
    state_values = [policy_value[state] for state in states]
    state_values_matrix = np.reshape(np.array(state_values), (18, 10))
    plt.imshow(state_values_matrix, cmap='hot', origin='lower', extent=((0,10,4,21)))
    plt.show()
    # visualize the other half of the states where usable_ace is False
    states = [state for state in generate_half_states(usable_ace=True)]
    state_values = [policy_value[state] for state in states]
    state_values_matrix = np.reshape(np.array(state_values), (18, 10))
    plt.imshow(state_values_matrix, cmap='hot', origin='lower', extent=((0,10,4,21)))
    plt.show()

# generic utilities ------------------------------------------------------------

# initialize a map over all states that maps every state to 0
def init_map_over_states():
    state_map = dict()
    for state in generate_states():
        state_map[state] = 0
    return state_map

# choose an action
def choose_action(policy, observation):
    # action_dict is a map from actions to probability of selection
    action_dict = policy[observation]
    action_space = action_dict.keys()
    p_distn = action_dict.values()
    return np.random.choice(action_space, 1, p=p_distn)[0]

# generate an episode represented as a list of tuples of form:
# (observation, action, reward, next_observation)
def generate_episode(env, policy):
    episode = list()
    observation = env.reset()
    while True:
        #print "observation: " + str(observation)
        action = choose_action(policy, observation)
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

# evaluate a policy by first-visit monte carlo policy evaluation
def policy_eval(env, policy, num_episodes=100000):
    # init state value function
    v = init_map_over_states()
    # init a map from states to the number of first_visits to that state
    visits_map = init_map_over_states()
    for _ in xrange(num_episodes):
        episode = generate_episode(env, policy)
        visited_states = set()
        n = len(episode)
        # keep track of cumulative return over the episode
        ret = 0
        # loop backwards through the episode
        for i in range(n-1, -1, -1):
            (state, _, reward, _) = episode[i]
            visited_states.add(state)
            avg_return = v[state]
            # compute cumulative return
            ret += reward
            m = visits_map[state] + 1
            # compute an updated average return
            update_avg_return = avg_return + (1.0 / m) * (ret - avg_return)
            v[state] = update_avg_return
        # update the counter for each state
        for state in visited_states:
            visits_map[state] += 1
        # increment the episode index
    return v

# main functionality -----------------------------------------------------------

def main():
    policy = init_policy()
    env = gym.make("Blackjack-v0")
    v = policy_eval(env, policy)
    visualize_value_function(v)

main()
