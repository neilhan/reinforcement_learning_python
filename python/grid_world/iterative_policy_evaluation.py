import numpy as np
import matplotlib.pyplot as plt
from grid_world.grid import Grid
from grid_world.utils import print_policy, print_values

SMALL_ENOUGH = 0.0001  # consider converged


def find_value_for_uniform_random_action_policy(grid: Grid):
    states = grid.get_all_states()

    # init v(s) = 0
    V = {}
    for s in states:
        V[s] = 0.0

    gamma = 1.0  # discount factor
    # repeat until convergence
    while True:
        biggest_change = 0
        for s in states:  # loop through all possible state
            old_v = V[s]
            if s in grid.action_map:  # not a terminal state
                new_v = 0.0
                p_a = 1.0 / len(grid.action_map[s])  # equal probability for each action
                for a in grid.action_map[s]:
                    grid.set_state(s)  # reset for a-loop
                    r = grid.move(a)
                    new_v = new_v + p_a * (r + gamma * V[grid.get_current_state()])
                V[s] = new_v
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))

        if biggest_change < SMALL_ENOUGH:
            break  # stop the value learning

    return V


def find_value_for_fixed_policy(grid: Grid, policy):
    '''
    policy: {(x,y): U/D/L/R, ...}
    '''
    states = grid.get_all_states()

    # init v(s) = 0
    V = {}
    for s in states:
        V[s] = 0.0

    # try a gamma
    gamma = 0.9  # discount

    # repeat until convergance
    while True:
        biggest_change = 0
        for s in states:
            old_v = V[s]
            if s in policy:  # not a terminal state
                a = policy[s]  # this time, not loop all possible actions, take the fixed action
                grid.set_state(s)
                r = grid.move(a)
                V[s] = r + gamma * V[grid.get_current_state()]
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))

        if biggest_change < SMALL_ENOUGH:
            break

    return V

def main():
    grid = Grid.build_standard_grid()

    # policy: uniformly random actions
    # ----------------------------------------------
    V = find_value_for_uniform_random_action_policy(grid)

    print('values for uniformly random actions:')
    print_values(V, grid)
    print('\n\n')

    # given a Fixed-policy
    # ----------------------------------------------
    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'U',
        (2, 1): 'R',
        (2, 2): 'U',
        (2, 3): 'L',
    }

    V = find_value_for_fixed_policy(grid, policy)

    print('giving policy:')
    print_policy(policy, grid)
    print('values for fixed policy:')
    print_values(V, grid)

if __name__ == '__main__':
    main()
