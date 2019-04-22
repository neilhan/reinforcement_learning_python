import numpy as np
from builtins import range
import matplotlib.pyplot as plt

from grid_world.grid import Grid
from grid_world.utils import print_policy, print_values

SMALL_ENOUGH = 0.0001  # consider converged
GAMMA = 0.9  # decay
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')


def apply_wind_to_action(a):
    """
    when policy decided to take 'a' action, giving the grid env is windy,
    0.5 chance that the agent ended with the rest 3 unintended actions.
    :param a:
    :return:
    """
    p = np.random.random()
    if p < 0.5:
        return a
    else:
        temp = list(ALL_POSSIBLE_ACTIONS)
        temp.remove(a)
        return np.random.choice(temp)


def play_game(grid: Grid, policy):
    """
    returns (states, corresponding returns)
    Needs reset game start at a random position, since policy is deterministic.
    If starting from the starting spot, we would not able to measure all states.
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
        a = apply_wind_to_action(a)  # apply wind
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

    # V init ----------------------------
    V = {}
    returns = {}
    states = grid.get_all_states()
    for s in states:
        if s in grid.action_map:
            returns[s] = []
        else:
            V[s] = 0.0

    # repeat game play
    for t in range(100):
        states_returns = play_game(grid, policy)
        seen_states = set()
        for s, G in states_returns:
            # seen s? if first visit, calc
            if s not in seen_states:
                returns[s].append(G)
                V[s] = np.mean(returns[s])
                seen_states.add(s)

    # look at the result
    print('found policy:')
    print_policy(policy, grid)
    print('found value:')
    print_values(V, grid)
