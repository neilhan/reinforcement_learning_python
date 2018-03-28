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


if __name__ == '__main__':
    unittest.main()


