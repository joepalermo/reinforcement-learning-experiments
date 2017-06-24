from Blackjack import Blackjack
from utilities import init_state_action_map, \
                      init_equiprobable_random_policy, \
                      generate_episode, \
                      off_policy_evaluation, \
                      greedy_stochastic_policy_improvement, \
                      fine_grained_off_policy_iteration

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

# perform episode-wise off-policy iteration
def policy_iteration(env, target_policy, behavior_policy):
    q = init_state_action_map(env)
    c = init_state_action_map(env)
    for _ in xrange(20000):
        episode = generate_episode(env, behavior_policy)
        off_policy_evaluation(episode, q, c, target_policy, behavior_policy)
        greedy_stochastic_policy_improvement(env, episode, q, target_policy)
    return q

def policy_iteration2(env, target_policy, behavior_policy):
    q = init_state_action_map(env)
    c = init_state_action_map(env)
    for _ in xrange(20000):
        episode = generate_episode(env, behavior_policy)
        fine_grained_off_policy_iteration(episode, q, c, target_policy, behavior_policy, gamma=1)
    return q

# main functionality -----------------------------------------------------------

def main():
    env = Blackjack()
    target_policy = init_policy(env)
    behavior_policy = init_equiprobable_random_policy(env)
    q = policy_iteration(env, target_policy, behavior_policy)
    env.visualize_action_value(q)

main()
