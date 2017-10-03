import random
import numpy as np

# A data structure to hold training steps for subsequent replay
# holds the most recent self.max_size training steps
class ReplayMemory:

    def __init__(self, max_size=1000):
        # once ReplayMemory is full, self.i determines where to add new values
        self.i = 0
        self.ls = []
        self.max_size = max_size

    def __getitem__(self, slice_obj):
        return self.ls[slice_obj]

    def __len__(self):
        return len(self.ls)

    def add(self, value):
        if len(self.ls) < self.max_size: # ReplayMemory isn't full
            self.ls.append(value)
        else:
            self.ls[self.i] = value
            self.i += 1
            if self.i == self.max_size:
                self.i = 0 # reset index back to 0 when it reaches max size

# select an epsilon greedy action
def choose_epsilon_greedy_action(q, encoded_state, epsilon):
    if random.random() < epsilon:
        return random.choice(range(q.num_actions))
    else:
        action_values = q.propagate(encoded_state)[0]
        return np.argmax(action_values)

# generate an episode using epsilon greedy actions
def generate_epsilon_greedy_episode(env, q, state_encoder, epsilon):
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
def estimate_performance(env, q, state_encoder, epsilon=0.1, num_episodes=25):
    episode_lengths = [len(generate_epsilon_greedy_episode(env, q, state_encoder, epsilon)) \
                       for _ in range(num_episodes)]
    avg = sum(episode_lengths) / num_episodes
    print("average episode length: {}".format(avg))
