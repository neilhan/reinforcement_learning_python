import numpy as np
from builtins import range
import matplotlib.pyplot as plt

from grid_world.grid import Grid
from grid_world.utils import print_policy, print_values

SMALL_ENOUGH = 0.0001  # consider converged
GAMMA = 0.9  # decay
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')
LEARNING_RATE = 0.001


def play_game(grid: Grid, policy):
    """
    returns (states, corresponding_returns)
    Needs reset game start at a random position, since policy is deterministic,
    we would not able to measure all states
    :param grid:
    :param policy:
    :return: (states, corresponding returns)
    """
    start_states = list(grid.action_map.keys())
    start_idx = np.random.choice(len(start_states))
    grid.set_state(start_states[start_idx])

    s = grid.get_current_state()
    states_rewards = [(s, 0.0)]
    while not grid.is_game_over():
        a = policy[s]
        r = grid.move(a)
        s = grid.get_current_state()
        states_rewards.append((s, r))

    # get returns from state_rewards
    G = 0
    states_returns = []
    is_first = True
    for s, r in reversed(states_rewards):
        if is_first:  # no need to calc the term state
            is_first = False
        else:
            states_returns.append((s, G))
        G = r + GAMMA * G

    states_returns.reverse()
    return states_returns


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
        (1, 2): 'U',
        (2, 1): 'L',
        (2, 2): 'U',
        (2, 3): 'L',
    }

    # have a look at the init-policy
    print('init policy:')
    print_policy(policy, grid)

    # theta
    theta = np.random.randn(4) / 2


    def s2x(s):
        return np.array([s[0] - 1, s[1] - 1.5, s[0] * s[1] - 3, 1])
        # state to x mapping.
        # x = [row, col, row*col, 1]   -> 1 for bias term


    # learn theta
    deltas = []
    t = 1.0
    for itr in range(20000):
        if itr % 100 == 0:
            t += 0.01
        alpha = LEARNING_RATE / t

        biggest_change = 0
        states_returns = play_game(grid, policy)
        seen_states = set()
        for s, G in states_returns:
            if s not in seen_states:
                old_theta = theta.copy()
                x = s2x(s)
                V_hat = theta.dot(x)
                theta += alpha * (G - V_hat) * x
                biggest_change = max(biggest_change, np.abs(old_theta - theta).sum())
                seen_states.add(s)
        deltas.append(biggest_change)

    plt.plot(deltas)
    plt.show()

    # V init ----------------------------
    V = {}
    states = grid.get_all_states()
    for s in states:
        if s in grid.action_map:
            V[s] = theta.dot(s2x(s))
        else:
            V[s] = 0.0

    # look at the result
    print('found policy:')
    print_policy(policy, grid)
    print('found value:')
    print_values(V, grid)
