from Blackjack import Blackjack
from utilities import init_state_map, \
                      choose_stochastic_action

# utilities --------------------------------------------------------------------

# initialize a policy that only sticks on 20 or 21
def init_policy(env):
    policy = dict()
    for state in env.generate_states():
        policy[state] = dict()
        player_sum = state[0]
        if player_sum < 20:
            policy[state][1] = 1.0
            policy[state][0] = 0
        else:
            policy[state][1] = 0
            policy[state][0] = 1.0
    return policy

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

# evaluate a policy by first-visit monte carlo policy evaluation
def policy_eval(env, policy, num_episodes=50000):
    # init state value function
    v = init_state_map(env)
    # init a map from states to the number of first_visits to that state
    visits_map = init_state_map(env)
    for _ in xrange(num_episodes):
        episode = generate_episode(env, policy)
        state_updates = dict()
        n = len(episode)
        # keep track of cumulative return over the episode
        ret = 0
        # loop backwards through the episode
        for i in range(n-1, -1, -1):
            (state, _, reward, _) = episode[i]
            avg_return = v[state]
            # update cumulative return
            ret += reward
            m = visits_map[state] + 1
            # compute an updated average return
            state_updates[state] = avg_return + (1.0 / m) * (ret - avg_return)
        # update the visit counter for each visited state
        for state in state_updates:
            v[state] = state_updates[state]
            visits_map[state] += 1
    return v

# main functionality -----------------------------------------------------------

def main():
    env = Blackjack()
    policy = init_policy(env)
    v = policy_eval(env, policy)
    env.visualize_state_value(v)

main()
