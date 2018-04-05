import numpy as np
from builtins import range
import matplotlib.pyplot as plt

from grid_world.grid import Grid
from grid_world.utils import print_policy, print_values

SMALL_ENOUGH = 0.0001  # consider converged
GAMMA = 0.9  # decay
ALPHA = 0.1  # learning rate
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


def apply_epsilon_to_action(a, eps=0.1):
    """
    This is part of policy eval, picking the next action.
    given 'a' with probability: 1-eps+ eps/4
    For the rest 3 actions, with eps/4 for each action
    :param a:
    :param eps:
    :return:
    """
    p = np.random.random()
    if p < (1 - eps):
        return a
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)


def max_dict(d):
    max_key = None
    max_val = float('-inf')
    for k, v in d.items():
        if v > max_val:
            max_val = v
            max_key = k
    return max_key, max_val


def play_game(grid: Grid, policy):
    """
    returns (states, corresponding_rewards)
    Needs reset game start at a random position, since policy is deterministic,
    we would not able to measure all states
    :param grid:
    :param policy:
    :return: (states, corresponding_rewards)
    """
    s = (2, 0)  # start from the starting cell
    grid.set_state(s)
    # s_t, r_t, the r_t is result of (s_t-1, a_t-1)
    states_actions_rewards = [(s, 0.0)]
    while not grid.is_game_over():
        a = policy[s]
        a = apply_epsilon_to_action(a)

        r = grid.move(a)
        s = grid.get_current_state()

        states_actions_rewards.append((s, r))

    return states_actions_rewards


if __name__ == '__main__':
    # grid = Grid.build_standard_grid()
    grid = Grid.build_negative_grid()

    # have a look at rewards
    print('rewards:')
    print_values(grid.reward_map, grid)

    # init policy ---------------------
    # policy = {}
    # for s in list(grid.action_map.keys()):
    #     policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

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

    V = {}
    states = grid.get_all_states()
    for s in states:
        V[s] = 0.0

    # repeat game play, until convergence
    for play in range(1000):

        states_rewards = play_game(grid, policy)
        # starting state to be ignored.
        # last (s,r) is terminal state,  don't care
        for t in range(len(states_rewards) -1):
            s, _ = states_rewards[t]
            s_next, r = states_rewards[t+1]
            V[s] = V[s] + ALPHA *(r + GAMMA * V[s_next] - V[s])


    # look at the result
    print('found policy:')
    print_policy(policy, grid)
    print('found value:')
    print_values(V, grid)
