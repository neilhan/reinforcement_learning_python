import numpy as np
from builtins import range
import matplotlib.pyplot as plt

from grid_world.grid import Grid
from grid_world.utils import print_policy, print_values

SMALL_ENOUGH = 0.0001  # consider converged
GAMMA = 0.9  # decay
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')
ALPHA = 0.001


class Model:
    def __init__(self):
        self.theta = np.random.randn(4) / 2

    def s2x(self, s):
        # x = [row, col, row*col, 1]   -> 1 for bias term
        return np.array([s[0] - 1, s[1] - 1.5, s[0] * s[1] - 3, 1])

    def predict(self, s):
        x = self.s2x(s)
        return self.theta.dot(x)

    def grad(self, s):
        return self.s2x(s)


def apply_epsilon(a, eps=0.1):
    # we'll use epsilon-soft to ensure all states are visited
    # what happens if you don't do this? i.e. eps=0
    p = np.random.random()
    if p < (1 - eps):
        return a
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)


def play_game(grid: Grid, policy):
    """
    returns (states, corresponding_rewards)
    Needs reset game start at a random position, since policy is deterministic,
    we would not able to measure all states
    :param grid:
    param policy:
    :return: (states, corresponding returns)
    """
    s = (2, 0)
    grid.set_state(s)
    s = grid.get_current_state()
    states_rewards = [(s, 0.0)]
    while not grid.is_game_over():
        a = policy[s]
        a = apply_epsilon(a)
        r = grid.move(a)
        s = grid.get_current_state()
        states_rewards.append((s, r))

    return states_rewards


if __name__ == '__main__':
    # grid = Grid.build_standard_grid()
    grid = Grid.build_negative_grid()

    # have a look at rewards
    print('rewards:')
    print_values(grid.reward_map, grid)

    # init policy ---------------------
    # given a Fixed-policy -----------------------
    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'R',
        (2, 1): 'R',
        (2, 2): 'R',
        (2, 3): 'U',
    }

    # have a look at the init-policy
    print('init policy:')
    print_policy(policy, grid)

    # learn theta
    model = Model()
    deltas = []
    k = 1.0
    for itr in range(2000):
        if itr % 10 == 0:
            k += 0.01
        alpha = ALPHA / k
        biggest_change = 0.0

        states_rewards = play_game(grid, policy)
        seen_states = set()
        for t in range(len(states_rewards) - 1):
            s, r_pre = states_rewards[t]
            s2, r = states_rewards[t + 1]

            old_theta = model.theta.copy()

            if grid.is_terminal(s2):
                target = r
            else:
                target = r + GAMMA * model.predict(s2)

            model.theta += alpha * (target - model.predict(s)) * model.grad(s)

            biggest_change = max(biggest_change, np.abs(old_theta - model.theta).sum())

        deltas.append(biggest_change)

    plt.plot(deltas)
    plt.show()

    # V init ----------------------------
    V = {}
    states = grid.get_all_states()
    for s in states:
        if s in grid.action_map:
            V[s] = model.predict(s)
        else:
            V[s] = 0.0

    # look at the result
    print('found policy:')
    print_policy(policy, grid)
    print('found value:')
    print_values(V, grid)
