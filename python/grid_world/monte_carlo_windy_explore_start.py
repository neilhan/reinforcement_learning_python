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
    returns (states, corresponding returns)
    Needs reset game start at a random position, since policy is deterministic,
    we would not able to measure all states
    :param grid:
    :param policy:
    :return: (states, corresponding returns)
    """
    # random starting point. each game play is an exploration.
    start_states = list(grid.action_map.keys())
    start_idx = np.random.choice(len(start_states))
    grid.set_state(start_states[start_idx])

    # (s_t, a_t, r_t) r_t is result of (s_t-1, a_t-1)
    s = grid.get_current_state()
    # pick random action
    a = np.random.choice(ALL_POSSIBLE_ACTIONS)
    states_actions_rewards = [(s, a, 0.0)]
    seen_states = set()
    while True:
        old_s = grid.get_current_state()
        r = grid.move(a)
        s = grid.get_current_state()

        if s in seen_states:
            # to avoid infinite episode, bumping in to the wall repeately
            states_actions_rewards.append((s, None, -100))
            break
        elif grid.is_game_over():
            states_actions_rewards.append((s, None, r))
            break
        else:
            a = policy[s]  # next action
            states_actions_rewards.append((s, a, r))
        seen_states.add(s)

    # get returns from state_rewards
    G = 0  # init G
    states_actions_returns = []
    is_first = True
    for s, a, r in reversed(states_actions_rewards):
        if is_first:  # no need to calc the term state
            is_first = False
        else:
            states_actions_returns.append((s, a, G))
        G = r + GAMMA * G  # G for t-1 step.

    states_actions_returns.reverse()
    return states_actions_returns


def main():
    # grid = Grid.build_standard_grid()
    grid = Grid.build_negative_grid()

    # have a look at rewards
    print('rewards:')
    print_values(grid.reward_map, grid)

    # init policy ---------------------
    # given a Fixed-policy
    policy = {}
    for s in list(grid.action_map.keys()):
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

    # have a look at the init-policy
    print('init policy:')
    print_policy(policy, grid)

    # init Q, returns,
    Q = {}  # Q[(s,a)]-> returns
    returns = {}  # state -> list_of_returns
    states = grid.get_all_states()
    for s in states:
        if s in grid.action_map:
            Q[s] = {}
            for a in ALL_POSSIBLE_ACTIONS:
                Q[s][a] = 0
                returns[(s, a)] = []

    # repeat game play, until convergence
    deltas = []  # for dev display
    for t in range(2000):
        if t % 100 == 0:
            print('game play:', str(t))

        biggest_change = 0
        states_actions_returns = play_game(grid, policy)
        seen_state_action_pairs = set()
        for s, a, G in states_actions_returns:
            # seen s? if first visit, calc
            sa = (s, a)
            if s not in seen_state_action_pairs:
                old_q = Q[s][a]
                returns[sa].append(G)
                Q[s][a] = np.mean(returns[sa])
                biggest_change = max(biggest_change, np.abs(old_q - Q[s][a]))
                seen_state_action_pairs.add(sa)
        deltas.append(biggest_change)
        # policy update after each game play
        for s in policy.keys():
            policy[s] = max_dict(Q[s])[0]

    plt.plot(deltas)
    plt.show()

    # just for printing, we have Q action-value function
    V = {}
    for s, Qs in Q.items():
        V[s] = max_dict(Q[s])[1]

    # look at the result
    print('found policy:')
    print_policy(policy, grid)
    print('found value:')
    print_values(V, grid)

if __name__ == '__main__':
    main()
