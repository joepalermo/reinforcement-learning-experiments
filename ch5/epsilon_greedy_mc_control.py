from Blackjack import Blackjack
from utilities import init_state_action_map, \
                      init_epsilon_greedy_policy, \
                      choose_stochastic_action

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

def policy_improvement(env, episode, q, policy, epsilon):
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

def policy_iteration(env, policy, epsilon):
    q = init_state_action_map(env)
    visits_map = init_state_action_map(env)
    for _ in xrange(50000):
        episode = generate_episode(env, policy)
        policy_evaluation(episode, q, visits_map)
        policy_improvement(env, episode, q, policy, epsilon)
    return q

def main():
    env = Blackjack()
    epsilon = 0.2
    policy = init_epsilon_greedy_policy(env, epsilon)
    q = policy_iteration(env, policy, epsilon)
    env.visualize_action_value(q)

main()
