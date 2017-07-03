import gym
from gym import spaces
from gym.utils import seeding

class RandomWalk(gym.Env):

    def __init__(self, n=5):
        self.n = n
        self.observation_space = spaces.Discrete(self.n)
        self.action_space = spaces.Discrete(2)
        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.position = (self.n - 1) / 2
        return self.position

    def _get_obs(self):
        return self.position

    def _step(self, action):
        assert self.action_space.contains(action)
        reward = 0
        done = False
        if action == 0:
            self.position -= 1
            if self.position == 0:
                done = True
        else:
            self.position += 1
            if self.position == self.n - 1:
                reward = 1
                done = True
        return self._get_obs(), reward, done, {}

    def generate_states(self):
        for state in xrange(self.n):
            yield state

    def generate_actions(self):
        for action in xrange(2):
            yield action

    def visualize_state_value(self, v):
        return [v[state] for state in self.generate_states()]
