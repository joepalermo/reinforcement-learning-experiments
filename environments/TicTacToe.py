import random
from collections import deque
from copy import deepcopy
from env_utilities import generate_probability_distribution
import numpy as np

class TicTacToe:
    def __init__(self):
        self.policy = init_policy()

    def reset(self):
        self.state = init_state()
        return self.state

    def get_obs(self):
        return self.state

    def step(self, action):
        # apply the move
        self.state = get_next_state(self.state, action, 'x')
        # check if move ends the game, and if so return as such
        is_terminal_state, winner = terminal_test(self.state)
        if winner == 'x':
            reward = 1
        else:
            reward = 0
        if is_terminal_state:
            return (self.state, reward, True, None)
        # otherwise apply the environment's move
        action_probs = self.policy[get_immutable_state(self.state)]
        action = get_action_with_prob(self.state, action_probs)
        # apply the move
        self.state = get_next_state(self.state, action, 'o')
        # check if the move ends the game and return
        is_terminal_state, winner = terminal_test(self.state)
        if winner == 'o':
            reward = -1
        else:
            reward = 0
        if is_terminal_state:
            done = True
        else:
            done = False
        return (self.state, reward, done, None)


# utilities --------------------------------------------------------------------

move_encoding = {(0,0):0, (0,1):1, (0,2):2, (1,0):3, (1,1):4, (1,2):5,
                 (2,0): 6, (2,1): 7, (2,2): 8}

move_decoding = {0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2),
                 6: (2,0), 7:(2,1), 8:(2,2)}

def encode_move(move):
    return move_encoding[move]

def decode_move(move_i):
    return move_decoding[move_i]

# return a representation of the initial game state (i.e. empty game board)
def init_state(immutable=False):
    if immutable:
        return ((None, None, None),(None, None, None),(None, None, None))
    else:
        return [[None, None, None],[None, None, None],[None, None, None]]

# get a list of valid moves
def get_valid_moves(state):
    all_positions = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
    return [(i,j) for (i,j) in all_positions if not state[i][j]]

def get_action_with_prob(state, action_probs):
    valid_moves = get_valid_moves(state)
    encoded_valid_moves = [encode_move(move) for move in valid_moves]
    encoded_move = np.random.choice(encoded_valid_moves, p=action_probs)
    return decode_move(encoded_move)

# get a random valid action
def get_random_action(state):
    valid_moves = get_valid_moves(state)
    move_index = random.randint(0,len(valid_moves)-1)
    return valid_moves[move_index]

def choose_greedy_action(value_function, state):
    greedy_action = (None, -float("inf"))
    for move in get_valid_moves(state):
        next_state = get_next_state(state, move, 'x')
        value = value_function[get_immutable_state(next_state)]
        if value > greedy_action[1]:
            greedy_action = (move, value)
    return greedy_action[0]

# convert a mutable representation of state to an immutable (and hashable) one
def get_immutable_state(state):
    return ((state[0][0], state[0][1], state[0][2]),
            (state[1][0], state[1][1], state[1][2]),
            (state[2][0], state[2][1], state[2][2]))

# return the state that follows after a given move
def get_next_state(state, move, turn_of):
    next_state = deepcopy(state)
    (i,j) = move
    next_state[i][j] = turn_of
    return next_state

def switch_player(turn_of):
    if turn_of is 'x':
        return 'o'
    else:
        return 'x'

# check if any game-ending conditions are satisfied
# the game ends if:
#   - there is a row, column, or diagonal containing 3 of only 1 kind of symbol
#    (i.e. a player has won)
#   - or, symbols have been placed in all 9 positions (i.e. tie game)
def terminal_test(state):
    is_terminal_state = False
    winner = get_winner(state)
    if winner or is_full_state(state):
        is_terminal_state = True
    return (is_terminal_state, winner)

# if a player_1 has won the game, return player_1
# if player_2 has won the game, return player_2
# else return None
def get_winner(state):
    winners_symbol = None
    # check the rows
    for i in range(0, len(state)):
        if check_squares_for_win(state[i][0], state[i][1], state[i][2]):
            winners_symbol = state[i][0]
    # check the columns
    for i in range(0, len(state)):
        if check_squares_for_win(state[0][i], state[1][i], state[2][i]):
            winners_symbol = state[0][i]
    # check the diagonals
    # top-left to bottom-right
    if check_squares_for_win(state[0][0], state[1][1], state[2][2]):
        winners_symbol = state[0][0]
    # top-right to bottom-left
    if check_squares_for_win(state[0][2], state[1][1], state[2][0]):
        winners_symbol = state[0][2]
    # if a winner has been found, return the winner's name
    return winners_symbol

# check to see if a sequence of squares have the same symbol
def check_squares_for_win(x, y, z):
    # make sure at least one value isn't None, and that all values are equal
    if x and x == y and y == z:
        return True
    return False

# check if the state is full with symbols
def is_full_state(state):
    for row in state:
        for square in row:
            if square is None:
                return False
    return True

# pretty-print the state
def print_state(state):
    print
    for i in range(0, 3):
        row_string = str(state[i][0]) + ' | ' + str(state[i][1]) + ' | ' + \
                     str(state[i][2])
        print(row_string.replace('None', ' '))

# initialize a value-function with 0.5 for all states except terminal ones
def init_value_function():
    value_function = dict()
    initial_state = init_state(immutable=False)
    frontier = deque([(initial_state, 'x')])
    while len(frontier) > 0:
        state, turn_of = frontier.popleft()
        is_terminal_state, winner = terminal_test(state)
        if is_terminal_state and winner is 'x':
            value_function[get_immutable_state(state)] = 1
        elif is_terminal_state and winner is 'o':
            value_function[get_immutable_state(state)] = 0
        elif is_terminal_state:
            value_function[get_immutable_state(state)] = 0.5
        else:
            value_function[get_immutable_state(state)] = 0.5
            # otherwise, get subsequent states and add them to the frontier
            for move in get_valid_moves(state):
                next_state = get_next_state(state, move, turn_of)
                frontier.append((next_state, switch_player(turn_of)))
    return value_function

# initialize a policy that maps each state to a random probability distribution
# over remaining valid actions
def init_policy():
    policy = dict()
    initial_state = init_state(immutable=False)
    frontier = deque([(initial_state, 'x')])
    while len(frontier) > 0:
        state, turn_of = frontier.popleft()
        n = len(get_valid_moves(state))
        dist = generate_probability_distribution(n)
        policy[get_immutable_state(state)] = dist
        # otherwise, get subsequent states and add them to the frontier
        for move in get_valid_moves(state):
            next_state = get_next_state(state, move, turn_of)
            is_terminal_state, _ = terminal_test(next_state)
            if not is_terminal_state:
                frontier.append((next_state, switch_player(turn_of)))
    return policy
