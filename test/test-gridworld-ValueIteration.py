import numpy as np

import sys
sys.path.append('src/')
from GridWorld import GridWorld
from ValueIteration import ValueIteration

def testGridworldValueIteration():
    grid = GridWorld('data/world00.csv', reward={0: -0.04, 1: 1.0, 2: -1.0, 3: np.NaN}, random_rate=0.2)

    solver = ValueIteration(grid.reward_function, grid.transition_model, gamma=0.9)
    solver.train()

    grid.visualize_value_policy(policy=solver.policy, values=solver.values)
    # grid.random_start_policy(policy=solver.policy, start_pos=(2, 0), n=1000)


if __name__ == "__main__":
    testGridworldValueIteration()