from Blackjack import Blackjack
from utilities import init_state_action_map, \
                      init_deterministic_policy, \
                      choose_deterministic_action, \
                      get_random_state_action

# utilities --------------------------------------------------------------------

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

# perform episode-wise policy iteration
def policy_iteration(env, policy):
    q = init_state_action_map(env)
    visits_map = init_state_action_map(env)
    for _ in xrange(50000):
        episode = generate_episode_es(env, policy)
        policy_evaluation(episode, q, visits_map)
        policy_improvement(env, episode, q, policy)
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

# perform deterministic policy improvement over all states in an episodes
def policy_improvement(env, episode, q, policy):
    for (state, _, _, _) in episode:
        actions = [action for action in env.generate_actions(state)]
        best_action = (-1, -float("inf"))
        for i, action in enumerate(actions):
            value = q[state][action]
            if value > best_action[1]:
                best_action = (i, value)
        best_action = actions[best_action[0]]
        policy[state] = best_action

# main functionality -----------------------------------------------------------

def main():
    env = Blackjack()
    policy = init_deterministic_policy(env)
    q = policy_iteration(env, policy)
    env.visualize_action_value(q)

main()
