from __future__ import print_function, absolute_import, division

import numpy as np
from tic_tac_toe import *
from tic_tac_toe.game_env import Environment


class Human:  # state
    def __init__(self):
        pass

    def set_player(self, player):
        """player is either env.x, or env.o. """
        self.player = player

    def take_action(self, env: Environment):
        while True:
            # take the action, and break the while loop
            # if move is legal
            print()
            move = input("Enter coordinates i, j for next move (i,j=0..2), for example: 1,2 : ")
            i, j = move.split(',')
            i = int(i)
            j = int(j)
            if env.is_empty_cell(i, j):
                env.board[i, j] = self.player
                break

    def update_V_after_episode(self, env):
        pass

    def update_state_history(self, s):
        pass
