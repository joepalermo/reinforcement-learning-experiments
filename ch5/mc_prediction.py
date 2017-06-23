from Blackjack import Blackjack
from utilities import init_state_map, \
                      generate_episode, \
                      on_policy_episode_state_evaluation

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

# main functionality -----------------------------------------------------------

def main():
    env = Blackjack()
    policy = init_policy(env)
    v = init_state_map(env)
    visits_map = init_state_map(env)
    for _ in xrange(20000):
        episode = generate_episode(env, policy)
        on_policy_episode_state_evaluation(episode, v, visits_map)
    env.visualize_state_value(v)

main()
