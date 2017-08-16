import random
import numpy as np

# select an epsilon greedy action
def choose_epsilon_greedy_action(q, encoded_state, epsilon):
    if random.random() < epsilon:
        return random.choice(range(q.num_outputs))
    else:
        action_values = q.propagate(encoded_state)[0]
        return np.argmax(action_values)

# generate an episode using epsilon greedy actions
def generate_epsilon_greedy_episode(env, q, state_encoder, epsilon=0.1):
    episode = []
    state = env.reset()
    done = False
    while not done:
        encoded_state = state_encoder(env, state)
        action = choose_epsilon_greedy_action(q, encoded_state, epsilon)
        (next_state, reward, done, _) = env.step(action)
        episode_step = (state, action, reward, next_state)
        episode.append(episode_step)
        state = next_state
    return episode

# determine post-training performance
def estimate_performance(env, q, state_encoder, num_episodes=10):
    episode_lengths = [len(generate_epsilon_greedy_episode(env, q, state_encoder)) \
                       for _ in range(num_episodes)]
    avg = sum(episode_lengths) / num_episodes
    print("average episode length: {}".format(avg))
