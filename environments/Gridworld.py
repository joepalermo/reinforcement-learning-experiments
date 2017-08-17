import random
import gym
from gym import spaces
from gym.utils import seeding
from grid_rendering import Viewer

class Gridworld(gym.Env):

    metadata = {
        'render.modes': ['human'],
    }

    def __init__(self, x_limit, y_limit, goals, anti_goals, kings_moves=False, wind=None, stochastic_wind=False):
        # set base attributes of the grid
        self.x_limit = x_limit
        self.y_limit = y_limit
        self.goals = goals
        # anti_goals are states with negative reward, i.e. the cliff
        self.anti_goals = anti_goals
        # set observation_space
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.x_limit),
            spaces.Discrete(self.y_limit)))
        # set action_space
        if kings_moves:
            self.action_space = spaces.Discrete(8)
        else:
            self.action_space = spaces.Discrete(4)
        # set wind, example: wind=[0,0,0,1,1,1,2,2,1,0]
        if wind:
            self.wind = wind
        else:
            self.wind = [0 for x in range(0, self.x_limit)]
        # set flags
        self.kings_moves = kings_moves
        self.stochastic_wind = stochastic_wind
        # reset state
        self._seed()
        self._reset()
        # set rendering object to None by default
        self.viewer = None
        # keep track of cumulative reward
        self.cumulative_reward = 0

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.coordinates = self.reset_coordinates()
        return self.coordinates

    def _get_obs(self):
        return self.coordinates

    def _step(self, action):
        assert self.action_space.contains(action)
        # allow moves in the pattern of a King from chess
        if self.kings_moves:
            if action == 0:
                self.apply_action((0, 1))
            elif action == 1:
                self.apply_action((1, 1))
            elif action == 2:
                self.apply_action((1, 0))
            elif action == 3:
                self.apply_action((1, -1))
            elif action == 4:
                self.apply_action((0, -1))
            elif action == 5:
                self.apply_action((-1, -1))
            elif action == 6:
                self.apply_action((-1, 0))
            elif action == 7:
                self.apply_action((-1, 1))
        # allow only vertical and horizontal moves
        else:
            if action == 0:
                self.apply_action((0, 1))
            elif action == 1:
                self.apply_action((1, 0))
            elif action == 2:
                self.apply_action((0, -1))
            elif action == 3:
                self.apply_action((-1, 0))
        # determine whether the goal state has been reached
        if self.coordinates in self.goals:
            reward = 0
            done = True
        elif self.coordinates in self.anti_goals:
            reward = -100
            done = False
            self.reset_coordinates()
        else:
            reward = -1
            done = False
        self.cumulative_reward += reward
        return self._get_obs(), reward, done, {}

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
        elif self.viewer is None:
            self.viewer = Viewer(self.x_limit, self.y_limit)
        else:
            entity_map = {'agent': self.coordinates,
                          'goal': self.goals,
                          'anti_goal': self.anti_goals}
            self.viewer.update(entity_map)

    def _close(self):
        if self.viewer is not None:
            self.viewer.close()

    # return a set of coordinates within the allowable range
    def reset_coordinates(self, fixed = (0,0)):
        if fixed:
            (x, y) = fixed
        else:
            x = self.np_random.randint(self.x_limit)
            y = self.np_random.randint(self.y_limit)
        return (x,y)

    # apply the effect of an action
    def apply_action(self, action):
        (x, y) = self.coordinates
        (delta_x, delta_y) = action
        # determine translation due to the wind
        if self.stochastic_wind:
            wind = random.choice([self.wind[x]-1, self.wind[x], self.wind[x]+1])
        else:
            wind = self.wind[x]
        # apply the effect of the action and the wind
        (x_, y_) = (x + delta_x, y + delta_y + wind)
        # ensure that x stays within bounds
        if 0 <= x_ and x_ <= self.x_limit-1:
            new_x = x_
        elif x_ <= self.x_limit-1:
            new_x = 0
        else:
            new_x = self.x_limit-1
        # ensure that y stays within bounds
        if 0 <= y_ and y_ < self.y_limit:
            new_y = y_
        elif y_ <= self.y_limit-1:
            new_y = 0
        else:
            new_y = self.y_limit-1
        self.coordinates = (new_x, new_y)

    # generate all states
    def generate_states(self):
        for x in range(self.x_limit):
            for y in range(self.y_limit):
                yield x, y

    # generate all actions from a given state
    def generate_actions(self, state):
        for x in range(0,self.action_space.n):
            yield x
