from __future__ import print_function, absolute_import, division

import numpy as np
import matplotlib.pyplot as plt
from tic_tac_toe.tic_tac_toe_env import Environment


def play_game(p1, p2, env, draw=False):
    # loops until game over
    current_player = None
    while not env.game_over():
        # alternate between players
        # p1 first
        if current_player == p1:
            current_player = p2
        else:
            current_player = p1

        # draw, for user to see
        if draw:
            if draw == 1 and current_player == p1:
                env.draw_board()
            if draw == 2 and current_player == p2:
                env.draw_board()

        # current_player move
        current_player.take_action(env)

        # update state_history
        state = env.get_state()
        p1.update_state_history(state)
        p2.update_state_history(state)

    if draw:
        env.draw_board()

    # value function update
    p1.update(env)
    p2.update(env)

