from Blackjack import Blackjack
from utilities import init_state_action_map, \
                      init_deterministic_policy, \
                      generate_episode_es, \
                      policy_eval_on_episode

# utilities --------------------------------------------------------------------

# perform deterministic policy improvement over all states in an episode
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

# perform episode-wise policy iteration
def policy_iteration(env, policy):
    q = init_state_action_map(env)
    visits_map = init_state_action_map(env)
    for _ in xrange(50000):
        episode = generate_episode_es(env, policy)
        on_policy_episode_eval(episode, q, visits_map)
        policy_improvement(env, episode, q, policy)
    return q

# main functionality -----------------------------------------------------------

def main():
    env = Blackjack()
    policy = init_deterministic_policy(env)
    q = policy_iteration(env, policy)
    env.visualize_action_value(q)

main()
