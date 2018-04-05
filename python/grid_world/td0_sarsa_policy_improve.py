import numpy as np
from builtins import range
import matplotlib.pyplot as plt

from grid_world.grid import Grid
from grid_world.utils import print_policy, print_values

GAMMA = 0.9  # decay
ALPHA = 0.1  # learning rate
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')


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

    # no policy, will derive from Q

    # init Q(s,a)
    Q = {}
    states = grid.get_all_states()
    for s in states:
        Q[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            Q[s][a] = 0.0

    # how many times Q[s] has been updated
    update_counts = {}
    update_counts_sa = {}
    for s in states:
        update_counts_sa[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            update_counts_sa[s][a] = 1.0

    # repeat game play, until convergence
    t = 1.0
    deltas = []
    for itr in range(20000):
        if itr % 100 == 0:
            t += 0.01
        if itr % 2000 == 0:
            print('itr:', itr)

        # not generate episoda, we play
        s = (2, 0)
        grid.set_state(s)

        # starting state to be ignored.
        # last (s,r) is terminal state,  don't care
        a = max_dict(Q[s])[0]
        a = apply_epsilon_to_action(a, eps=0.5 / t)
        biggest_change = 0
        while not grid.is_game_over():
            r = grid.move(a)
            s2 = grid.get_current_state()

            a2 = max_dict(Q[s2])[0]
            a2 = apply_epsilon_to_action(a2, eps=0.5 / t)

            alpha = ALPHA / update_counts_sa[s][a]
            update_counts_sa[s][a] += 0.005
            old_qsa = Q[s][a]
            Q[s][a] = old_qsa + alpha * (r + GAMMA * Q[s2][a2] - old_qsa)
            biggest_change = max(biggest_change, np.abs(old_qsa - Q[s][a]))

            update_counts[s] = update_counts.get(s, 0) + 1

            # next
            s = s2
            a = a2

        deltas.append(biggest_change)

    plt.plot(deltas)
    plt.show()

    policy = {}
    V = {}
    for s in grid.action_map.keys():
        a, max_q = max_dict(Q[s])
        policy[s] = a
        V[s] = max_q

    # look at the result
    print('found policy:')
    print_policy(policy, grid)
    print('found value:')
    print_values(V, grid)
