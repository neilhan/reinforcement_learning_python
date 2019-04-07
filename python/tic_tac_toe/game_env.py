from __future__ import print_function, absolute_import, division

import numpy as np
import matplotlib.pyplot as plt
from tic_tac_toe import BOARD_LENGTH


class Environment:  # state
    def __init__(self):
        self.board = np.zeros([BOARD_LENGTH, BOARD_LENGTH])
        self.x = -1  # x piece on the board
        self.o = 1  # o piece on the board
        self.winner = None
        self.ended = False
        self.num_states = 3 ** (BOARD_LENGTH * BOARD_LENGTH)

    def get_state(self):
        """
        Return current_state, represented as an int
        from 0...|S|-1, where S is the set of all states
        |S| = 3^board_size.
        Since each cell has 3 possible value, empty, x, o, this is integer in base-3 number.
        c1, c2, c3,
        c4, c5, c6
        c7, c8, c9
        c1*3^0 + c2 * 3^1 + c3 * 3^2 ... c9 * 3^8
        :return:
        """
        k = 0
        h = 0
        for i in range(BOARD_LENGTH):
            for j in range(BOARD_LENGTH):
                if self.board[i, j] == 0:
                    v = 0
                elif self.board[i, j] == self.x:
                    v = 1
                elif self.board[i, j] == self.o:
                    v = 2
                h += (3 ** k) * v
                k += 1
        return h

    def _player_win(self, player) -> None:
        """
        :param player self.x or self.o
        """
        self.winner = player
        self.ended = True

    def is_game_over(self, force_recalc=False):
        """
        return True if game over, player won or draw.
        This also set winner, ended fields. (not a good design)
        :param force_recalc:
        :return:
        """
        if not force_recalc and self.ended:
            return self.ended

        # any rows
        for i in range(BOARD_LENGTH):
            for player in (self.x, self.o):
                if self.board[i].sum() == player * BOARD_LENGTH:
                    self._player_win(player)
                    return True
        # any columns
        for j in range(BOARD_LENGTH):
            for player in (self.x, self.o):
                if self.board[:, j].sum() == player * BOARD_LENGTH:
                    self._player_win(player)
                    return True

        # diagonals
        for player in (self.x, self.o):
            # top-left -> bottom-right diagonal
            if self.board.trace() == player * BOARD_LENGTH:
                self._player_win(player)
                return True
            # top-right -> bottom-left diagonal
            if np.fliplr(self.board).trace() == player * BOARD_LENGTH:
                self.winner = player
                return True

        # is draw, all cells are not 0
        if np.all((self.board == 0) == False):
            self._player_win(None)
            return True

        # not game over
        self.winner = None
        return False

    def is_empty_cell(self, i, j):
        return self.board[i, j] == 0

    def get_empty_cells(self):
        empty_cells = []
        for i in range(BOARD_LENGTH):
            for j in range(BOARD_LENGTH):
                if self.is_empty_cell(i, j):
                    empty_cells.append((i, j))

        return empty_cells

    def get_reward(self, player):
        if not self.is_game_over():
            return 0

        return 1 if player == self.winner else 0

    def get_board_str(self):
        result = ''
        for i in range(BOARD_LENGTH):
            result += '-------------' + '\n'
            for j in range(BOARD_LENGTH):
                result += '| '
                if self.board[i, j] == self.x:
                    result += 'x '
                elif self.board[i, j] == self.o:
                    result += 'o '
                else:
                    result += '  '

            result += '|\n'

        result += '-------------' + '\n'

        return result

    def draw_board(self):
        board_str = self.get_board_str()
        print(board_str)
