import numpy as np
import matplotlib.pyplot as plt

class PolicyIteration:
    def __init__(self, reward_function, transition_model, gamma, init_policy=None, init_value=None) -> None:
        self.num_states = transition_model.shape[0]
        self.num_actions = transition_model.shape[1]

        self.reward_function = np.nan_to_num(reward_function)
        self.transition_model = transition_model
        self.gamma = gamma
        
        if init_policy is None:
            self.policy = np.random.randint(0, self.num_actions, self.num_states)
        else:
            self.policy = init_policy
        
        self.values = np.zeros(self.num_states)

    def one_policy_evaluation(self):
        delta = 0
        for s in np.arange(self.num_states):
            curr_utility = self.values[s]
            a = self.policy[s]                  # Action a taken under policy π at state s
            p = self.transition_model[s, a]     # Transition matrix P to yield s' for action a taken under policy π at state s
            self.values[s] = self.reward_function[s] + self.gamma * np.sum(p * self.values) # Bellman update for state utility
            delta = max(0, abs(curr_utility - self.values[s])) # Maximum difference in utility, across all states, for a given policy π after policy evaluation
        return delta

    def policy_evaluation(self, convergence_threshold=1e-3):
        epoch = 0
        delta = self.one_policy_evaluation()
        delta_history = [delta]
        while epoch < 500:
            delta = self.one_policy_evaluation()
            delta_history.append(delta)
            if delta < convergence_threshold:
                break
            epoch += 1
        return len(delta_history)

    def policy_improvement(self):
        update_policy_count = 0
        for s in np.arange(self.num_states):
            curr_action = self.policy[s]                                        # Current action a taken under policy π at state s
            successor_utilities = np.zeros(self.num_actions)                  # Expected values of choosing UP, RIGHT, DOWN, LEFT from state s given transition function P
            for a in np.arange(self.num_actions):
                p = self.transition_model[s, a]                                 # Transition matrix P to yield s' for action a taken under policy π at state s
                successor_utilities[a] = np.sum(p * self.values)              # Bellman update of expected value choosing UP, RIGHT, DOWN, LEFT
            self.policy[s] = np.argmax(successor_utilities)                   # Update policy at state s with action a which yields the max expected utility (stochastic environment)
            update_policy_count += 1 if self.policy[s] != curr_action else 0    # Count how many actions were changed in policy after policy improvement
        return update_policy_count

    def train(self, convergence_threshold=1e-3, plot=True):
        epoch = 0
        policy_evaluation_count = self.policy_evaluation(convergence_threshold)
        policy_evaluation_count_hist = [policy_evaluation_count]
        policy_improvement_count = self.policy_improvement()
        policy_improvement_count_hist = [policy_improvement_count]
        while epoch < 500:
            epoch += 1
            new_policy_evaluation_count = self.policy_evaluation(convergence_threshold)
            policy_evaluation_count_hist.append(new_policy_evaluation_count)
            new_policy_improvement_count = self.policy_improvement()
            policy_improvement_count_hist.append(new_policy_improvement_count)
            
            # If policy doesn't change, our policy has converged. Optimal policy.
            if new_policy_improvement_count == 0:
                break

        print(f'# epoch: {len(policy_improvement_count_hist)}')
        print(f'eval count = {policy_evaluation_count_hist}')
        print(f'policy change = {policy_improvement_count_hist}')

        if plot is True:
            fig, axes = plt.subplots(2, 1, figsize=(3.5, 4), sharex='all', dpi=200)
            axes[0].plot(np.arange(len(policy_evaluation_count_hist)), policy_evaluation_count_hist, marker='o', markersize=4, alpha=0.7,
                         color='#2ca02c', label='# sweep in \npolicy evaluation\n' + r'$\gamma =$' + f'{self.gamma}')
            axes[0].legend()

            axes[1].plot(np.arange(len(policy_improvement_count_hist)), policy_improvement_count_hist, marker='o',
                         markersize=4, alpha=0.7, color='#d62728',
                         label='# policy updates in \npolicy improvement\n' + r'$\gamma =$' + f'{self.gamma}')
            axes[1].set_xlabel('Epoch')
            axes[1].legend()
            plt.tight_layout()
            plt.show()