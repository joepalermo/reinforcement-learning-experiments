from Blackjack import Blackjack
from utilities import init_state_action_map, \
                      init_epsilon_greedy_policy, \
                      generate_episode, \
                      on_policy_episode_evaluation, \
                      epsilon_greedy_policy_improvement

# perform episode-wise on-policy iteration for an epsilon greedy policy
def policy_iteration(env, policy, epsilon):
    q = init_state_action_map(env)
    visits_map = init_state_action_map(env)
    for _ in xrange(50000):
        episode = generate_episode(env, policy)
        on_policy_episode_evaluation(episode, q, visits_map)
        epsilon_greedy_policy_improvement(env, episode, q, policy, epsilon)
    return q

# main functionality -----------------------------------------------------------

def main():
    env = Blackjack()
    epsilon = 0.4
    policy = init_epsilon_greedy_policy(env, epsilon)
    q = policy_iteration(env, policy, epsilon)
    env.visualize_action_value(q)

main()
