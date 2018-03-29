from __future__ import print_function, absolute_import, division

from tic_tac_toe.game_env import Environment

import unittest


class TestEnvironment(unittest.TestCase):

    def test_get_state(self):
        env = Environment()
        state = env.get_state()
        self.assertTrue(state == 0)

        env.board[0, 1] = env.x
        state = env.get_state()
        self.assertTrue(state == 3)

        env.board[0, 1] = env.o
        state = env.get_state()
        self.assertTrue(state == 6)

        env.board[0, 2] = env.x
        state = env.get_state()
        self.assertTrue(state == 15)

    def test_draw_board(self):
        env = Environment()
        env.draw_board()
        expected = \
            '-------------\n' + \
            '|   |   |   |\n' + \
            '-------------\n' + \
            '|   |   |   |\n' + \
            '-------------\n' + \
            '|   |   |   |\n' + \
            '-------------\n'
        self.assertTrue(env.get_board_str() == expected)

        env.board[1, 1] = -1
        env.board[1, 2] = 1
        env.draw_board()
        expected = \
            '-------------\n' + \
            '|   |   |   |\n' + \
            '-------------\n' + \
            '|   | x | o |\n' + \
            '-------------\n' + \
            '|   |   |   |\n' + \
            '-------------\n'

        self.assertTrue(env.get_board_str() == expected)


if __name__ == '__main__':
    unittest.main()
