from Blackjack import Blackjack
from utilities import init_state_action_map, \
                      init_deterministic_policy, \
                      generate_episode_es, \
                      on_policy_evaluation, \
                      greedy_deterministic_policy_improvement

# perform episode-wise on-policy iteration with exploring starts
def policy_iteration(env, policy):
    q = init_state_action_map(env)
    visits_map = init_state_action_map(env)
    for _ in xrange(20000):
        episode = generate_episode_es(env, policy)
        on_policy_evaluation(episode, q, visits_map)
        greedy_deterministic_policy_improvement(env, episode, q, policy)
    return q

# main functionality -----------------------------------------------------------

def main():
    env = Blackjack()
    policy = init_deterministic_policy(env)
    q = policy_iteration(env, policy)
    env.visualize_action_value(q)

main()
