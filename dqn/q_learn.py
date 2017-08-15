import sys
from os.path import abspath, join, dirname
# add the top level package to sys.path to access them
sys.path.insert(1, abspath(join(dirname(__file__), '../environments')))

import time
import random
from Qnet import Qnet
from GridworldChase import GridworldChase, state_encoder
from dqn_utils import max_q, choose_epsilon_greedy_action, estimate_performance
import numpy as np

# construct a mini-batch from the replay memory
def construct_mini_batch(q, replay_memory, mbs, gamma):
    n = len(replay_memory)
    num_actions = q.env.action_space.n
    random_indices = [random.randint(0,n-1) for _ in range(mbs)]
    raw_mb = [replay_memory[i] for i in random_indices]
    xs, ys, actions = [], [], []
    for (state, action, reward, next_state) in raw_mb:
        # construct x
        x = state_encoder(q.env, state, dim=3)
        xs.append(x)
        # construct y
        y = np.zeros(num_actions)
        x = state_encoder(q.env, state)
        xp = state_encoder(q.env, next_state)
        y[action] = reward + gamma * max_q(q, xp) - q.propagate(x)[action]
        ys.append(y)
        # construct a
        a = np.zeros(num_actions)
        a[action] = 1
        actions.append(a)
    return np.array(xs), np.array(ys), np.array(actions)

# learn a q-network
def q_network_learning(env, q, num_episodes=10, mbs=100, epsilon=0.1, gamma=1, eta=0.1):
    replay_memory = []
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
            replay_memory.append(step_data)
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
    num_episodes = 50
    mbs = 100
    epsilon = 0.1
    gamma = 1
    eta = 0.1

    # create an env
    env = GridworldChase(10, 10)

    # initialize a q-network
    q = Qnet(env)

    # estimate its performance against the environment
    estimate_performance(env, q, state_encoder)

    # train it
    q_network_learning(env, q, num_episodes=num_episodes, mbs=100, epsilon=epsilon, gamma=gamma, eta=eta)

    # estimate its performance against the environment
    estimate_performance(env, q, state_encoder)


    # demonstrate learning
    state = env.reset()
    while True:
        env.render()
        time.sleep(0.25)
        encoded_state = state_encoder(env, state)
        action = choose_epsilon_greedy_action(q, encoded_state, epsilon)
        state, reward, done, _ = env.step(action)
        print(state, reward, done)
        if done:
            env.render(close=True)
            break


main()
