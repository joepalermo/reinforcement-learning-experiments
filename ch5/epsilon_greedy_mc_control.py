from Blackjack import Blackjack
from utilities import init_state_action_map, \
                      init_epsilon_greedy_policy, \
                      generate_episode, \
                      policy_eval_on_episode

# utilities --------------------------------------------------------------------

# perform epsilon-greedy policy improvement over all states in an episode
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

# perform episode-wise on-policy iteration
def policy_iteration(env, policy, epsilon):
    q = init_state_action_map(env)
    visits_map = init_state_action_map(env)
    for _ in xrange(50000):
        episode = generate_episode(env, policy)
        on_policy_episode_eval(episode, q, visits_map)
        policy_improvement(env, episode, q, policy, epsilon)
    return q

# main functionality -----------------------------------------------------------

def main():
    env = Blackjack()
    epsilon = 0.4
    policy = init_epsilon_greedy_policy(env, epsilon)
    q = policy_iteration(env, policy, epsilon)
    env.visualize_action_value(q)

main()
