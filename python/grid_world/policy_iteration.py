import numpy as np
from builtins import range
import matplotlib.pyplot as plt

from grid_world.grid import Grid
from grid_world.utils import print_policy, print_values

SMALL_ENOUGH = 0.0001  # consider converged
GAMMA = 0.9  # decay
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

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
        # policy evaluation
        while True:
            biggest_change = 0
            for s in policy:
                old_v = V[s]
                a = policy[s]
                grid.set_state(s)
                r = grid.move(a)
                V[s] = r + GAMMA * V[grid.get_current_state()]
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))

            if biggest_change < SMALL_ENOUGH:
                break  # break while for policy-eval

        # policy update?
        is_policy_converged = True
        for s in policy:
            old_a = policy[s]
            new_a = old_a
            best_value = float('-inf')
            # find best for this state
            for a in ALL_POSSIBLE_ACTIONS:
                grid.set_state(s)
                r = grid.move(a)
                next_state = grid.get_current_state()
                v = r + GAMMA * V[next_state]
                if v > best_value:
                    best_value = v
                    new_a = a
            policy[s] = new_a
            if old_a != policy[s]:
                is_policy_converged = False

        if is_policy_converged:
            break  # break policy outer loop

        itr_count += 1
        print('iteration:' + str(itr_count))

    # look at the result
    print('found policy:')
    print_policy(policy, grid)
    print('found value:')
    print_values(V, grid)
