import numpy as np
from builtins import range
import matplotlib.pyplot as plt

from grid_world.grid import Grid
from grid_world.utils import print_policy, print_values

SMALL_ENOUGH = 0.0001  # consider converged
GAMMA = 0.9  # decay
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')
ALPHA = 0.001

SA2IDX = {}
IDX = 0


class Model:
    def __init__(self):
        self.theta = np.random.randn(25) / np.sqrt(25)

    def sa2x(self, s, a):
        # x = [row, col, row*col, , row^2, col^2, 1]   for each action
        return np.array([
            s[0] - 1 if a == 'U' else 0,
            s[1] - 1.5 if a == 'U' else 0,
            (s[0] * s[1] - 3) / 3 if a == 'U' else 0,
            (s[0] * s[0] - 2) / 2 if a == 'U' else 0,
            (s[1] * s[1] - 4.5) / 4.5 if a == 'U' else 0,
            1 if a == 'U' else 0,
            s[0] - 1 if a == 'D' else 0,
            s[1] - 1.5 if a == 'D' else 0,
            (s[0] * s[1] - 3) / 3 if a == 'D' else 0,
            (s[0] * s[0] - 2) / 2 if a == 'D' else 0,
            (s[1] * s[1] - 4.5) / 4.5 if a == 'D' else 0,
            1 if a == 'D' else 0,
            s[0] - 1 if a == 'L' else 0,
            s[1] - 1.5 if a == 'L' else 0,
            (s[0] * s[1] - 3) / 3 if a == 'L' else 0,
            (s[0] * s[0] - 2) / 2 if a == 'L' else 0,
            (s[1] * s[1] - 4.5) / 4.5 if a == 'L' else 0,
            1 if a == 'L' else 0,
            s[0] - 1 if a == 'R' else 0,
            s[1] - 1.5 if a == 'R' else 0,
            (s[0] * s[1] - 3) / 3 if a == 'R' else 0,
            (s[0] * s[0] - 2) / 2 if a == 'R' else 0,
            (s[1] * s[1] - 4.5) / 4.5 if a == 'R' else 0,
            1 if a == 'R' else 0,
            1])

    def predict(self, s, a):
        x = self.sa2x(s, a)
        return self.theta.dot(x)

    def grad(self, s, a):
        return self.sa2x(s, a)

    def getQs(self, s):
        Qs = {}
        for a in ALL_POSSIBLE_ACTIONS:
            q_sa = model.predict(s, a)
            Qs[a] = q_sa
        return Qs


def max_dict(d):
    max_key = None
    max_val = float('-inf')
    for k, v in d.items():
        if v > max_val:
            max_val = v
            max_key = k
    return max_key, max_val


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

    states = grid.get_all_states()
    for s in states:
        SA2IDX[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            SA2IDX[s][a] = IDX
            IDX += 1

    model = Model()

    # learn theta
    deltas = []
    t = 1.0
    t2 = 1.0
    for itr in range(20000):
        biggest_change = 0.0

        if itr % 1000 == 0:
            print('itr:', itr)

        if itr % 10 == 0:
            t += 0.001
            t2 += 0.008 

        alpha = ALPHA / t2

        # play_game
        s = (2, 0)
        grid.set_state(s)
        Qs = model.getQs(s)

        a = max_dict(Qs)[0]
        a = apply_epsilon(a, eps=0.5 / t)  # epsilon greedy

        while not grid.is_game_over():
            r = grid.move(a)
            s2 = grid.get_current_state()

            old_theta = model.theta.copy()
            if grid.is_terminal(s2):
                model.theta += alpha * (r - model.predict(s, a)) * model.grad(s, a)
            else:  # not terminal state, continue game
                Qs2 = model.getQs(s2)
                a2 = max_dict(Qs2)[0]
                a2 = apply_epsilon(a2, eps=0.5 / t)

                # update Q
                model.theta += alpha * (r + GAMMA * model.predict(s2, a2) - model.predict(s, a)) * model.grad(s, a)

                # next step
                s = s2
                a = a2

            biggest_change = max(biggest_change, np.abs(old_theta - model.theta).sum())

        deltas.append(biggest_change)

    plt.plot(deltas)
    plt.show()

    policy = {}
    V = {}
    Q = {}
    for s in grid.action_map.keys():
        Qs = model.getQs(s)
        Q[s] = Qs
        a, max_q = max_dict(Qs)
        policy[s] = a
        V[s] = max_q

    # look at the result
    print('found policy:')
    print_policy(policy, grid)
    print('found value:')
    print_values(V, grid)
