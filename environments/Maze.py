import random
import gym
from gym import spaces
from gym.utils import seeding
from grid_rendering import Viewer

class Maze(gym.Env):

    metadata = {
        'render.modes': ['human'],
    }

    def __init__(self, x_limit, y_limit, goals, walls, kings_moves=False):
        # set base attributes of the maze
        self.x_limit = x_limit
        self.y_limit = y_limit
        self.goals = goals
        # anti_goals are states with negative reward, i.e. the cliff
        self.walls = walls
        # set observation_space
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.x_limit),
            spaces.Discrete(self.y_limit)))
        # set action_space
        if kings_moves:
            self.action_space = spaces.Discrete(8)
        else:
            self.action_space = spaces.Discrete(4)
        # set flags
        self.kings_moves = kings_moves
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
                          'wall': self.walls}
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
        new_position = (x + delta_x, y + delta_y)
        # only apply action if new position is within the env's boundaries
        if self.position_within_boundary(new_position):
            self.coordinates = new_position

    def position_within_boundary(self, position):
        (x, y) = position
        if 0 <= x and x <= self.x_limit-1:
            if 0 <= y and y <= self.y_limit-1:
                if (x,y) not in self.walls:
                    return True
        return False

    # generate all states
    def generate_states(self):
        for x in xrange(self.x_limit):
            for y in xrange(self.y_limit):
                yield x, y

    # generate all actions from a given state
    def generate_actions(self, state):
        for x in xrange(0,self.action_space.n):
            yield x
