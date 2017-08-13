from gym.envs.toy_text.blackjack import BlackjackEnv
import numpy as np
import matplotlib
# setup matplotlib
matplotlib.use('TkAgg')
# import plot function
import matplotlib.pyplot as plt

# wrapper class for gym's BlackjackEnv
class Blackjack(BlackjackEnv):

    # generate all states
    def generate_states(self):
        for player_sum in range(4,22):
            for dealer_showing in range(1,11):
                for usable_ace in [False, True]:
                    yield (player_sum, dealer_showing, usable_ace)

    # generate all states with the supplied value of usable_ace
    def generate_half_states(self, usable_ace=False):
        for player_sum in range(4,22):
            for dealer_showing in range(1,11):
                yield (player_sum, dealer_showing, usable_ace)

    # generate all actions from a given state
    def generate_actions(self, state):
        for action in range(0,2):
            yield action

    # visualize the action value function as a heatmap
    def visualize_action_value(self, q):
        # visualize action value where usable_ace is False and action is hit
        states = [state for state in self.generate_half_states(usable_ace=False)]
        state_values = [q[state][1] for state in states]
        state_values_matrix = np.reshape(np.array(state_values), (18, 10))
        plt.imshow(state_values_matrix, cmap='hot', origin='lower', extent=((0,10,4,21)))
        plt.show()
        # visualize action value where usable_ace is True and action is hit
        states = [state for state in self.generate_half_states(usable_ace=True)]
        state_values = [q[state][1] for state in states]
        state_values_matrix = np.reshape(np.array(state_values), (18, 10))
        plt.imshow(state_values_matrix, cmap='hot', origin='lower', extent=((0,10,4,21)))
        plt.show()
        # visualize action value where usable_ace is False and action is stick
        states = [state for state in self.generate_half_states(usable_ace=False)]
        state_values = [q[state][0] for state in states]
        state_values_matrix = np.reshape(np.array(state_values), (18, 10))
        plt.imshow(state_values_matrix, cmap='hot', origin='lower', extent=((0,10,4,21)))
        plt.show()
        # visualize action value where usable_ace is True and action is stick
        states = [state for state in self.generate_half_states(usable_ace=True)]
        state_values = [q[state][0] for state in states]
        state_values_matrix = np.reshape(np.array(state_values), (18, 10))
        plt.imshow(state_values_matrix, cmap='hot', origin='lower', extent=((0,10,4,21)))
        plt.show()


    # visualize the state value function as a heatmap
    def visualize_state_value(self, v):
        # visualize state value where usable_ace is False
        states = [state for state in self.generate_half_states(usable_ace=False)]
        state_values = [v[state] for state in states]
        state_values_matrix = np.reshape(np.array(state_values), (18, 10))
        plt.imshow(state_values_matrix, cmap='hot', origin='lower', extent=((0,10,4,21)))
        plt.show()
        # visualize state value where usable_ace is True
        states = [state for state in self.generate_half_states(usable_ace=True)]
        state_values = [v[state] for state in states]
        state_values_matrix = np.reshape(np.array(state_values), (18, 10))
        plt.imshow(state_values_matrix, cmap='hot', origin='lower', extent=((0,10,4,21)))
        plt.show()
