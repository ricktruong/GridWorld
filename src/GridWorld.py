import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from time import time

class GridWorld:
    def __init__(self, filename, reward, random_rate, time_limit=1000) -> None:
        file = open(filename)
        self.map = np.array([list(map(float, line.strip().split(','))) for line in file.readlines()])
        file.close()
        self.reward = reward
        self.random_rate = random_rate
        self.time_limit = time_limit
        self.num_actions = 4

        self.num_rows = self.map.shape[0]
        self.num_cols = self.map.shape[1]
        self.num_states = self.num_rows * self.num_cols

        self.reward_function = self.get_reward_function()
        self.transition_model = self.get_transition_model()

    def get_state_from_pos(self, pos):
        return pos[0] * self.num_cols + pos[1]
    
    def get_pos_from_state(self, state):
        return state // self.num_cols, state % self.num_rows
    
    def get_reward_function(self):
        reward_table = np.zeros(self.num_states)
        for r in np.arange(self.num_rows):
            for c in np.arange(self.num_cols):
                s = self.get_state_from_pos((r, c))
                reward_table[s] = self.reward[self.map[r, c]]
        return reward_table

    def get_transition_model(self):
        transitional_model = np.zeros((self.num_states, self.num_actions, self.num_states))
        for r in np.arange(self.num_rows):
            for c in np.arange(self.num_cols):
                s = self.get_state_from_pos((r, c))
                next_s = np.zeros(self.num_actions)
                # If current position is an empty square, determine next states
                if self.map[r, c] == 0:
                    for action in np.arange(self.num_actions):
                        next_r, next_c = r, c
                        # Up
                        if action == 0:
                            next_r = max(0, r - 1)
                        # Right
                        elif action == 1:
                            next_c = min(self.num_cols - 1, c + 1)
                        # Down
                        elif action == 2:
                            next_r = min(self.num_rows - 1, r + 1)
                        # Left
                        elif action == 3:
                            next_c = max(0, c - 1)

                        # If next state is the wall from this action, next state is this same state
                        if self.map[next_r, next_c] == 3:
                            next_r, next_c = r, c
                        
                        s_prime = self.get_state_from_pos((next_r, next_c))
                        next_s[action] = s_prime

                # If current position is the wall, diamond, or poison, next states are current state
                else:
                    next_s = np.ones(self.num_actions) * s
                    
                # Calculate transition model for each action
                for action in np.arange(self.num_actions):
                    transitional_model[s, action, int(next_s[action])] += 1 - self.random_rate                              # P(s') += 0.8 when taking action a at state s
                    transitional_model[s, action, int(next_s[(action + 1) % self.num_actions])] += self.random_rate / 2.0   # P(s'+right) += 0.1 when taking action a at state s
                    transitional_model[s, action, int(next_s[(action - 1) % self.num_actions])] += self.random_rate / 2.0   #P(s'+left) += 0.1 when taking action a at state s
        return transitional_model
    
    def generate_random_policy(self):
        return np.random.randint(self.num_actions, size=self.num_states)
    
    def execute_policy(self, policy, start_pos=(2, 0)):
        s = self.get_state_from_pos(start_pos)
        total_reward = 0
        state_history = [s]
        while True:
            temp_transition = self.transition_model[s, policy[s]]           # State transition matrix P for state s with action a given policy π
            s_prime = np.random.choice(self.num_states, p=temp_transition)  # Next state s' given state s, action a from policy π, and state transition matrix P (Stochastic decision-making)
            state_history.append(s_prime)
            
            r = self.reward_function[s_prime]   # Calculate reward of next state
            total_reward += r                   # Add reward to total reward
            s = s_prime                         # Move to next state

            # If reach diamond or poison, end GridWorld game
            if r == 1 or r == -1:
                break
            
        return total_reward
    
    def visualize_value_policy(self, policy, values, fig_size=(8, 6)):
        unit = min(fig_size[1] // self.num_rows, fig_size[0] // self.num_cols)
        unit = max(1, unit)
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        ax.axis('off')

        for i in range(self.num_cols + 1):
            if i == 0 or i == self.num_cols:
                ax.plot([i * unit, i * unit], [0, self.num_rows * unit],
                        color='black')
            else:
                ax.plot([i * unit, i * unit], [0, self.num_rows * unit],
                        alpha=0.7, color='grey', linestyle='dashed')
        for i in range(self.num_rows + 1):
            if i == 0 or i == self.num_rows:
                ax.plot([0, self.num_cols * unit], [i * unit, i * unit],
                        color='black')
            else:
                ax.plot([0, self.num_cols * unit], [i * unit, i * unit],
                        alpha=0.7, color='grey', linestyle='dashed')

        for i in range(self.num_rows):
            for j in range(self.num_cols):
                y = (self.num_rows - 1 - i) * unit
                x = j * unit
                s = self.get_state_from_pos((i, j))
                if self.map[i, j] == 3:
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='black',
                                             alpha=0.6)
                    ax.add_patch(rect)
                elif self.map[i, j] == 2:
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='red',
                                             alpha=0.6)
                    ax.add_patch(rect)
                elif self.map[i, j] == 1:
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='green',
                                             alpha=0.6)
                    ax.add_patch(rect)
                if self.map[i, j] != 3:
                    ax.text(x + 0.5 * unit, y + 0.5 * unit, f'{values[s]:.4f}',
                            horizontalalignment='center', verticalalignment='center',
                            fontsize=max(fig_size)*unit*0.6)
                if policy is not None:
                    if self.map[i, j] == 0:
                        a = policy[s]
                        symbol = ['^', '>', 'v', '<']
                        ax.plot([x + 0.5 * unit], [y + 0.5 * unit], marker=symbol[a], alpha=0.4,
                                linestyle='none', markersize=max(fig_size)*unit, color='#1f77b4')

        plt.tight_layout()
        plt.show()

    def random_start_policy(self, policy, start_pos, n=100, plot=True):
        start_time = int(round(time() * 1000))
        overtime = False
        scores = np.zeros(n)
        i = 0
        while i < n:
            temp = self.execute_policy(policy=policy, start_pos=start_pos)
            # print(f'i = {i} Random start result: {temp}')
            if temp > float('-inf'):
                scores[i] = temp
                i += 1
            cur_time = int(round(time() * 1000)) - start_time
            if cur_time > n * self.time_limit:
                overtime = True
                break

        print(f'max = {np.max(scores)}')
        print(f'min = {np.min(scores)}')
        print(f'mean = {np.mean(scores)}')
        print(f'std = {np.std(scores)}')

        if overtime is False and plot is True:
            bins = 100
            fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
            ax.set_xlabel('Total rewards in a single game')
            ax.set_ylabel('Frequency')
            ax.hist(scores, bins=bins, color='#1f77b4', edgecolor='black')
            plt.show()

        if overtime is True:
            print('Overtime!')
            return None
        else:
            return np.max(scores), np.min(scores), np.mean(scores)
        
    
    def __str__(self) -> str:
        return f"""
{self.map}
Reward: {self.reward}
Number of rows, columns, total states: {self.num_rows}, {self.num_cols}, {self.num_states}
Reward function: {self.reward_function}
Transition model: {self.transition_model}
Reward of one play of Gridworld through random policy: {self.execute_policy(self.generate_random_policy())}
"""