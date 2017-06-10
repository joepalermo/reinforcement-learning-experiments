from gym.envs.toy_text.blackjack import BlackjackEnv
import numpy as np
import matplotlib
# setup matplotlib
matplotlib.use('TkAgg')
# import plot function
import matplotlib.pyplot as plt

class Blackjack(BlackjackEnv):

    # generate all states
    def generate_states(self):
        for player_sum in xrange(4,22):
            for dealer_showing in xrange(1,11):
                for usable_ace in [False, True]:
                    yield (player_sum, dealer_showing, usable_ace)

    # generate all states with the supplied value of usable_ace
    def generate_half_states(self, usable_ace=False):
        for player_sum in xrange(4,22):
            for dealer_showing in xrange(1,11):
                yield (player_sum, dealer_showing, usable_ace)

    # generate all actions from a given state
    def generate_actions(self):
        for action in xrange(0,2):
            yield action

    # visualize the policy by way of the corresponding state-value function
    def visualize_policy(self, q):
        # visualize states where usable_ace is False
        states = [state for state in self.generate_half_states(usable_ace=False)]
        state_values = [q[state][1] for state in states]
        state_values_matrix = np.reshape(np.array(state_values), (18, 10))
        plt.imshow(state_values_matrix, cmap='hot', origin='lower', extent=((0,10,4,21)))
        plt.show()
        # visualize the other half of the states where usable_ace is False
        states = [state for state in self.generate_half_states(usable_ace=True)]
        state_values = [q[state][1] for state in states]
        state_values_matrix = np.reshape(np.array(state_values), (18, 10))
        plt.imshow(state_values_matrix, cmap='hot', origin='lower', extent=((0,10,4,21)))
        plt.show()


    # takes as input a value function (represented as a dictionary mapping states
    # to expected value) and displays a visualization of the value function as a
    # heatmap
    def visualize_value_function(self, policy_value):
        # visualize states where usable_ace is False
        states = [state for state in self.generate_half_states(usable_ace=False)]
        state_values = [policy_value[state] for state in states]
        state_values_matrix = np.reshape(np.array(state_values), (18, 10))
        plt.imshow(state_values_matrix, cmap='hot', origin='lower', extent=((0,10,4,21)))
        plt.show()
        # visualize the other half of the states where usable_ace is False
        states = [state for state in self.generate_half_states(usable_ace=True)]
        state_values = [policy_value[state] for state in states]
        state_values_matrix = np.reshape(np.array(state_values), (18, 10))
        plt.imshow(state_values_matrix, cmap='hot', origin='lower', extent=((0,10,4,21)))
        plt.show()
