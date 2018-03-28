from __future__ import print_function, absolute_import, division

from tic_tac_toe.game_env import Environment
from tic_tac_toe import play

import unittest

class TestPlay(unittest.TestCase):

    def test_win_state(self):
        env = Environment()
        all_term_states = play.get_state_hash_and_winner(env)
        self.assertTrue(len(all_term_states) == 3**9)


if __name__ == '__main__':
    unittest.main()


