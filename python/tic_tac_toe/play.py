from __future__ import print_function, absolute_import, division

import numpy as np
import matplotlib.pyplot as plt
from tic_tac_toe import *
from tic_tac_toe.game_env import Environment
from tic_tac_toe.agent import Agent
from tic_tac_toe.human import Human


def play_game(p1, p2, env: Environment, draw=False) -> None:
    # loops until game over
    current_player = None
    while not env.is_game_over():
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

    # after one game play, see who won?
    if draw:
        env.draw_board()

    # value function update
    p1.update_V_after_episode(env)
    p2.update_V_after_episode(env)


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
        if j >= (board_width - 1):
            if i >= (board_height - 1):
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


def _build_V_for(player, env, state_winner_triples):
    """
    returns the V lookup table: state->v as numpy array
    if player_x wins, V(s) = 1
    if x loses or draw, V(s) = 0
    otherwise, V(s) = 0.5
    :param env:  Environment
    :param state_winner_triples: [(state, winner, ended), (..), ..]
    :return:  V funciton as lookup table: state->v
    """
    V = np.zeros(env.num_states)
    for state, winner, ended in state_winner_triples:
        if ended:
            if winner == player:  # either player_x, or player_o
                v = 1
            else:
                v = 0
        else:
            v = 0.5
        V[state] = v
    return V


def build_init_V_for_player_x(env, state_winner_triples):
    return _build_V_for(env.x, env, state_winner_triples)


def build_init_V_for_player_o(env, state_winner_triples):
    return _build_V_for(env.o, env, state_winner_triples)


def main():
    env = Environment()
    state_winner_triples = get_state_hash_and_winner(env)

    player_x = Agent()
    V_x = build_init_V_for_player_x(env, state_winner_triples)
    player_x.set_V(V_x)
    player_x.set_player(env.x)

    player_o = Agent()
    V_o = build_init_V_for_player_o(env, state_winner_triples)
    player_o.set_V(V_o)
    player_o.set_player(env.o)

    # play games
    T = 10000
    for t in range(T):
        if t % 200 == 0:
            print(t)
        play_game(player_x, player_o, Environment())

    # human vs ai
    human = Human()
    human.set_player(env.o)
    while True:
        player_x.set_verbose(True)
        play_game(player_x, human, Environment(), draw=2)
        answer = input('Play again? Y/n: ')
        if answer and answer.lower()[0] == 'n':
            break


if __name__ == '__main__':
    main()
