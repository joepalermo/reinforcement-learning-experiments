import sys
from os.path import abspath, join, dirname
# add the top level package to sys.path to access them
sys.path.insert(1, abspath(join(dirname(__file__), '../environments')))

import time
import random
import numpy as np
from Qnet import Qnet
from GridworldChase import GridworldChase, state_encoder
from dqn_utils import ReplayMemory, \
                      choose_epsilon_greedy_action, \
                      estimate_performance

# construct a mini-batch from the replay memory
def construct_mini_batch(q, replay_memory, mbs, gamma):
    n = len(replay_memory)
    num_actions = q.env.action_space.n
    random_indices = [random.randint(0,n-1) for _ in range(mbs)]
    raw_mb = [replay_memory[i] for i in random_indices]
    xs, xps, actions, rewards = [], [], [], []
    for i, (state, action, reward, next_state) in enumerate(raw_mb):
        # construct x
        x = state_encoder(q.env, state)
        xs.append(x[0])
        # construct xp
        xp = state_encoder(q.env, next_state)
        xps.append(xp[0])
        # construct a
        a = np.zeros(num_actions)
        a[action] = 1
        actions.append(a)
        # construct reward
        rewards.append(reward)
    xs, xps, actions, rewards = np.array(xs), np.array(xps), np.array(actions), np.array(rewards)
    # construct ys
    ys_base = actions.copy()
    a_indices = np.argmax(actions, axis=1)
    ys_values = rewards + gamma * np.max(q.propagate(xps), axis=1) - q.propagate(xs)[np.arange(mbs), a_indices]
    ys_values = np.reshape(ys_values, (mbs, 1))
    ys = ys_base * ys_values
    return xs, ys, actions

# learn a q-network
def q_network_learning(env, q, num_episodes=10, mbs=100, epsilon=0.1, gamma=1, eta=0.1):
    replay_memory = ReplayMemory()
    for i in range(num_episodes):
        print(i)
        state = env.reset()
        done = False
        while not done:
            encoded_state = state_encoder(env, state)
            action = choose_epsilon_greedy_action(q, encoded_state, epsilon)
            (next_state, reward, done, _) = env.step(action)
            # store step data in replay memory
            step_data = (state, action, reward, next_state)
            replay_memory.add(step_data)
            # train on a mini-batch extracted from replay_memory
            if len(replay_memory) > mbs:
                xs, ys, actions = construct_mini_batch(q, replay_memory, mbs, gamma)
                q.sess.run(q.train_step, feed_dict={q.x: xs, \
                                                    q.y_: ys, \
                                                    q.a: actions,
                                                    q.eta : eta})
            state = next_state

def main():
    # define hyperparameters
    num_episodes = 40
    mbs = 128
    epsilon = 0.5
    gamma = 0.9
    eta = 0.5

    # create an env
    env = GridworldChase(8, 8, p_goal_move=0.5, goal_random_start=True)

    # initialize a q-network
    q = Qnet(env)

    # estimate its performance against the environment
    estimate_performance(env, q, state_encoder, epsilon=1)

    # train it
    q_network_learning(env, q, num_episodes, mbs, epsilon, gamma, eta)

    # estimate its performance against the environment
    estimate_performance(env, q, state_encoder, epsilon=0.05)

    # demonstrate learning
    state = env.reset()
    while True:
        env.render()
        time.sleep(0.25)
        encoded_state = state_encoder(env, state)
        action = choose_epsilon_greedy_action(q, encoded_state, 0.1)
        state, reward, done, _ = env.step(action)
        if done:
            env.render(close=True)
            break

if __name__ == '__main__':
    main()
