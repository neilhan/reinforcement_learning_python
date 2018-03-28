from __future__ import print_function, absolute_import, division

import numpy as np
import matplotlib.pyplot as plt
from tic_tac_toe import *
from tic_tac_toe.game_env import Environment


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


def get_state_hash_and_winner(env, i=0, j=0, board_width=BOARD_LENGTH, board_height=BOARD_LENGTH):
    """
    Return all states(as ints) and who is winner for those states if any.
    (i,j) refers to the next cell on the board to permute(need to try -1,0,1)
    impossible games are ignored. ie 3x and 3o in a row in one game, since that will never happen
    in real game.
    :param env: the env
    :param i: loop i start with
    :param j: loop j start with
    :return: [(state, winner, ended)]
    """
    results = []

    for v in (0, env.x, env.o):
        env.board[i, j] = v
        if j >= board_width:
            if i >= board_height:
                state = env.get_state()
                ended = env.is_game_over(force_recalc=True)
                winner = env.winner
                # only append to result when all 9 cells are filled
                results.append((state, winner, ended))
            else:
                results += get_state_hash_and_winner(env, i + 1, 0)  # reset j, go to next row
        else:
            results += get_state_hash_and_winner(env, i, j + 1)

    return results
