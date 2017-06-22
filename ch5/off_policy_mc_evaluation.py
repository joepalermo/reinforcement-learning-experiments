from Blackjack import Blackjack
from utilities import init_state_action_map, \
                      init_equiprobable_random_policy, \
                      generate_episode

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

def importance_sampling(target_policy, behavior_policy, state, action):
    return target_policy[state][action] / behavior_policy[state][action]

def policy_eval(env, target_policy, behavior_policy, gamma=1, num_episodes=100000):
    q = init_state_action_map(env)
    c = init_state_action_map(env)
    for _ in xrange(num_episodes):
        g = 0
        w = 1.0
        episode = generate_episode(env, behavior_policy)
        num_steps = len(episode)
        for i in xrange(num_steps-1, -1, -1):
            (state, action, reward, next_state) = episode[i]
            g = gamma * g + reward
            c[state][action] += w
            q[state][action] += w / c[state][action] * (g - q[state][action])
            w *= importance_sampling(target_policy, behavior_policy, state, action)
            if w == 0:
                break
    return q

def main():
    env = Blackjack()
    target_policy = init_policy(env)
    behavior_policy = init_equiprobable_random_policy(env)
    q = policy_eval(env, target_policy, behavior_policy)
    env.visualize_action_value(q)

main()
