import numpy as np
import matplotlib.pyplot as plt

class ValueIteration():
    def __init__(self, reward_function, transition_model, gamma) -> None:
        self.num_states = transition_model.shape[0]
        self.num_actions = transition_model.shape[1]

        self.reward_function = np.nan_to_num(reward_function)
        self.transition_model = transition_model
        self.gamma = gamma

        self.values = np.zeros(self.num_states)
        self.policy = None

    def one_iteration(self):
        delta = 0
        for s in np.arange(self.num_states):
            curr_utility = self.values[s]                       # Utility of current state s
            successor_utilities = np.zeros(self.num_actions)    # Utilities of successor states
            for a in np.arange(self.num_actions):
                p = self.transition_model[s, a]                 # Transition matrix P when taking action a in state s
                successor_utilities[a] = self.reward_function[s] + self.gamma * np.sum(p * self.values) # Bellman update is max of expected utilities across all successor states + immediate reward of current state
            self.values[s] = max(successor_utilities)
            delta = max(delta, abs(self.values[s] - curr_utility))  # Maximum difference in utility, across all states, after an iteration of value iteration
        return delta
    
    def get_policy(self):
        pi = np.ones(self.num_states) * -1
        for s in np.arange(self.num_states):
            successor_utilities = np.zeros(self.num_actions)
            for a in np.arange(self.num_actions):
                p = self.transition_model[s, a]
                successor_utilities[a] = np.sum(p * self.values)
            pi[s] = np.argmax(successor_utilities)
        return pi.astype(int)

    def train(self, covergence_threshold=1e-3, plot=True):
        epoch = 0
        delta = self.one_iteration()
        delta_history = [delta]
        while delta > covergence_threshold:
            epoch += 1
            delta = self.one_iteration()
            delta_history.append(delta)
        self.policy = self.get_policy()

        print(f'# iterations of policy improvement: {len(delta_history)}')
        print(f'delta = {delta_history}')

        if plot is True:
            fig, ax = plt.subplots(1, 1, figsize=(3, 2), dpi=200)
            ax.plot(np.arange(len(delta_history)) + 1, delta_history, marker='o', markersize=4,
                    alpha=0.7, color='#2ca02c', label=r'$\gamma= $' + f'{self.gamma}')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Delta')
            ax.legend()
            plt.tight_layout()
            plt.show()
