import numpy as np
from builtins import range
import matplotlib.pyplot as plt

from grid_world.grid import Grid
from grid_world.utils import print_policy, print_values

SMALL_ENOUGH = 0.0001  # consider converged
GAMMA = 0.9  # decay
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')


def policy_eval(grid: Grid, V, policy):
    while True:
        biggest_change = 0
        for s in policy:
            old_v = V[s]
            p_a = policy[s]
            new_v = 0.0
            # consider windy_grid. success possibility 0.5, other 3 direction: 0.5/3
            for a in ALL_POSSIBLE_ACTIONS:
                if a == p_a:
                    p = 0.5
                else:
                    p = 0.5 / 3
                grid.set_state(s)
                r = grid.move(a)
                new_v = new_v + p * (r + GAMMA * V[grid.get_current_state()])

            V[s] = new_v
            biggest_change = max(biggest_change, np.abs(old_v - V[s]))

        if biggest_change < SMALL_ENOUGH:
            break  # break while for policy-eval

    return V


def policy_improve(grid: Grid, V, policy):
    is_policy_converged = True
    for s in policy:
        old_a = policy[s]
        new_a = old_a
        best_value = float('-inf')
        # find best for this state
        for p_a in ALL_POSSIBLE_ACTIONS:
            # this is Windy-Grid. for any action, the result is Random.
            # 0.5 as expected, 0.5 chance to go to the rest 3 directions randomly.
            v = 0.0  # given an action, get the value for that action
            for a in ALL_POSSIBLE_ACTIONS:
                if a == p_a:
                    p = 0.5
                else:
                    p = 0.5 / 3
                grid.set_state(s)
                r = grid.move(a)
                next_state = grid.get_current_state()
                v = v + p * (r + GAMMA * V[next_state])
            if v > best_value:
                best_value = v
                new_a = p_a
        policy[s] = new_a
        if old_a != policy[s]:
            is_policy_converged = False

    return policy, is_policy_converged

    pass


if __name__ == '__main__':
    # grid = Grid.build_standard_grid()
    grid = Grid.build_negative_grid()

    # have a look at rewards
    print('rewards:')
    print_values(grid.reward_map, grid)

    # init policy ---------------------
    policy = {}
    for s in grid.action_map.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

    # have a look at the init-policy
    print('init policy:')
    print_policy(policy, grid)

    # V init ----------------------------
    V = {}
    states = grid.get_all_states()
    for s in states:
        if s in grid.action_map:
            V[s] = np.random.random()
        else:
            V[s] = 0.0

    itr_count = 0
    # policy search loop ------------------
    while True:  # break when policy-stopped-changing
        # policy evaluation, update value function
        V = policy_eval(grid, V, policy)

        # policy update? update policy
        policy, is_policy_converged = policy_improve(grid, V, policy)

        if is_policy_converged:
            break  # break policy outer loop

        itr_count += 1
        print('iteration:' + str(itr_count))

    # look at the result
    print('found policy:')
    print_policy(policy, grid)
    print('found value:')
    print_values(V, grid)
