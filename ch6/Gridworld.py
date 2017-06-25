import gym
from gym import spaces
from gym.utils import seeding

class Gridworld(gym.Env):

    def __init__(self):
        self.x_limit = 10
        self.y_limit = 7
        self.goal = (7,4)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.x_limit),
            spaces.Discrete(self.y_limit)))
        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.coordinates = self.reset_coordinates()
        return self._get_obs()

    def _step(self, action):
        assert self.action_space.contains(action)
        (x,y) = self.coordinates
        # set next state on the basis of action
        if action == 0 and y+1 <= self.y_limit-1:
            self.coordinates = (x, y+1)
        elif action == 1 and x+1 <= self.x_limit-1:
            self.coordinates = (x+1, y)
        elif action == 2 and y-1 >= 0:
            self.coordinates = (x, y-1)
        elif action == 3 and x-1 >= 0:
            self.coordinates = (x-1, y)
        # determine whether the goal state has been reached
        if self.coordinates == self.goal:
            reward = 0
            done = True
        else:
            reward = -1
            done = False
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return self.coordinates

    def reset_coordinates(self):
        x = self.np_random.randint(self.x_limit)
        y = self.np_random.randint(self.y_limit)
        return (x,y)

    def generate_states(self):
        for x in xrange(self.x_limit):
            for y in xrange(self.y_limit):
                yield x, y

    def generate_actions(self, state):
        for x in xrange(0,4):
            yield x
