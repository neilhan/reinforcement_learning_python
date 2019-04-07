from __future__ import print_function, absolute_import, division

import numpy as np
from tic_tac_toe import BOARD_LENGTH
from tic_tac_toe.game_env import Environment


class Agent:  # state
    def __init__(self, eps=0.1, alpha=0.5):
        self.eps = eps
        self.alpha = alpha
        self.verbose = False
        self.state_history = []

    def set_V(self, V):
        self.V = V

    def set_player(self, player):
        self.player = player

    def set_verbose(self, vb):
        self.verbose = vb

    def reset_history(self):
        self.state_history = []

    def take_action(self, env):
        # epsilon-greedy strategy, take random action if random value < epsilon
        r = np.random.rand()

        possible_moves = env.get_empty_cells()

        if r < self.eps:
            # act random, explore
            if self.verbose:
                print('random action')

            idx = np.random.choice(len(possible_moves))
            next_move = possible_moves[idx]
        else:  # take the best known most rewarding action
            value_map = {}  # debugging info

            next_move = None
            best_value = -1
            for i, j in possible_moves:
                env.board[i, j] = self.player
                state = env.get_state()
                env.board[i, j] = 0
                value_map[(i, j)] = self.V[state]
                if self.V[state] > best_value:
                    best_value = self.V[state]
                    next_move = (i, j)

        self._show_decision_making(env, value_map)

        # make the move
        env.board[next_move[0], next_move[1]] = self.player

    def update_state_history(self, s):
        self.state_history.append(s)

    def update_V_after_episode(self, env: Environment):
        """
        Happens after each game play episode.
        V(s) = V(s) + alpha*(V(next_s) - V(s))
        :param env:
        :return:
        """
        # get the reward. Since game over
        reward = env.get_reward(self.player)
        next_t_v = reward
        for s in reversed(self.state_history):
            v = self.V[s] + self.alpha * (next_t_v - self.V[s])
            self.V[s] = v
            next_t_v = v  # set v for next loop step

        self.reset_history()

    def _show_decision_making(self, env: Environment, value_map):
        if self.verbose:
            print("Taking a greedy action")
            for i in range(BOARD_LENGTH):
                print("------------------")
                for j in range(BOARD_LENGTH):
                    if env.is_empty_cell(i, j):
                        # print the value
                        print(" %.2f|" % value_map[(i, j)], end="")
                    else:
                        print("  ", end="")
                        if env.board[i, j] == env.x:
                            print("x  |", end="")
                        elif env.board[i, j] == env.o:
                            print("o  |", end="")
                        else:
                            print("   |", end="")
                print("")
            print("------------------")
