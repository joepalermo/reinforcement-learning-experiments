import random
import itertools
import gym
from gym import spaces
from gym.utils import seeding
from grid_rendering import Viewer
import numpy as np

def state_encoder(env, state, dim=4):
    (agent_x, agent_y) = state[0], state[1]
    (goal_x, goal_y) = state[2], state[3]
    if dim == 3:
        x = np.zeros((env.x_lim, env.y_lim, 2))
        x[agent_x, agent_y, 0] = 1
        x[goal_x, goal_y, 1] = 1
    elif dim == 4:
        x = np.zeros((1, env.x_lim, env.y_lim, 2))
        x[0, agent_x, agent_y, 0] = 1
        x[0, goal_x, goal_y, 1] = 1
    return x

class GridworldChase(gym.Env):

    metadata = {
        'render.modes': ['human'],
    }

    def __init__(self, x_lim, y_lim, walls=[], kings_moves=False):
        # set base attributes of the maze
        self.x_lim = x_lim
        self.y_lim = y_lim
        # walls block travel
        self.walls = walls
        # set observation_space
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.x_lim),
            spaces.Discrete(self.y_lim),
            spaces.Discrete(self.x_lim),
            spaces.Discrete(self.y_lim)))
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
        self.agent = self.reset_coordinates()
        self.goal = self.reset_coordinates(fixed=(self.x_lim-1, self.y_lim-1))
        self.state = {'agent': self.agent, 'goal': self.goal}
        return self._get_obs()

    def _get_obs(self):
        agent_x, agent_y = self.state['agent']
        goal_x, goal_y = self.state['goal']
        state = (agent_x, agent_y, goal_x, goal_y)
        return state

    def _step(self, action):
        assert self.action_space.contains(action)
        # move the agent
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
        # move the goal
        goal_action = random.choice(range(5))
        if goal_action == 0:
            self.apply_action((0, 1), entity='goal')
        elif goal_action == 1:
            self.apply_action((1, 0), entity='goal')
        elif goal_action == 2:
            self.apply_action((0, -1), entity='goal')
        elif goal_action == 3:
            self.apply_action((-1, 0), entity='goal')
        # null action
        elif goal_action == 4:
            pass
        # determine whether the goal state has been reached
        if self.state['agent'] == self.state['goal']:
            reward = 1
            done = True
        else:
            reward = 0
            done = False
        self.cumulative_reward += reward
        return self._get_obs(), reward, done, {}

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
        elif self.viewer is None:
            self.viewer = Viewer(self.x_lim, self.y_lim)
        else:
            entity_map = {'agent': self.state['agent'],
                          'goal': self.state['goal'],
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
            x = self.np_random.randint(self.x_lim)
            y = self.np_random.randint(self.y_lim)
        return (x,y)

    # apply the effect of an action
    def apply_action(self, action, entity='agent'):
        (x, y) = self.state[entity]
        (delta_x, delta_y) = action
        new_position = (x + delta_x, y + delta_y)
        # only apply action if new position is within the env's boundaries
        if self.position_within_boundary(new_position):
            self.state[entity] = new_position

    # check whether a set of coordinates is within the environment's boundaries
    def position_within_boundary(self, position):
        (x, y) = position
        if 0 <= x and x <= self.x_lim-1:
            if 0 <= y and y <= self.y_lim-1:
                if (x,y) not in self.walls:
                    return True
        return False

    # generate all states
    def generate_states(self):
        x_range = range(self.x_lim)
        y_range = range(self.y_lim)
        for (agent_x, agent_y) in itertools.product(x_range,y_range):
            for (goal_x, goal_y) in itertools.product(x_range,y_range):
                yield (agent_x, agent_y, goal_x, goal_y)

    # generate all actions from a given state
    def generate_actions(self, state):
        for x in range(0,self.action_space.n):
            yield x
