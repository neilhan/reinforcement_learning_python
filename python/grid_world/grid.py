from __future__ import print_function, absolute_import, division

import numpy as np
import matplotlib.pyplot as plt


class Grid:
    def __init__(self, height: int, width: int, start: (int, int)):
        self.width = width
        self.height = height
        self.current_state = start

    def set(self, reward_map, action_map):
        """
        :param reward: rewards is stored as a dict of: (i, j)-> r
        :param actions: actions should be a dict of: (i, j): A. A is list of actions
        :return: None
        """
        self.reward_map = reward_map
        self.action_map = action_map

    def set_state(self, s: (int, int)):
        self.current_state = s

    def get_current_state(self) -> (int, int):
        return self.current_state

    def is_terminal(self, s):
        return s not in self.action_map

    def move(self, action: chr) -> float:
        """
        Return reward. The grid:
        (0,0) (0,1) (0,2) ..
        (1,0) (1,1) (1,2) ..
        (2,0) (2,1) (2,2) ..
        .. .. .. ..
        :param action:
        :return:
        """
        # is legal?
        if action in self.action_map[self.get_current_state()]:
            y, x = self.current_state
            if action == 'u' or action == 'U':
                self.set_state((y - 1, x))
            elif action == 'd' or action == 'D':
                self.set_state((y + 1, x))
            elif action == 'l' or action == 'L':
                self.set_state((y, x - 1))
            elif action == 'r' or action == 'R':
                self.set_state((y, x + 1))

        return self.reward_map.get(self.current_state, 0)

    def undo_move(self, action):
        """
        This is to undo the move.
        :param action:
        :return:
        """
        y, x = self.current_state
        if action == 'u' or action == 'U':
            self.set_state((y + 1, x))
        elif action == 'd' or action == 'D':
            self.set_state((y - 1, x))
        elif action == 'l' or action == 'L':
            self.set_state((y, x + 1))
        elif action == 'r' or action == 'R':
            self.set_state((y, x - 1))

        assert (self.current_state in self.get_all_states())

    def get_all_states(self):
        return set(list(self.reward_map.keys()) + list(self.action_map.keys()))

    @staticmethod
    def build_standard_grid():
        """
        builder. x - means can't go to that cell. s - means start cell
        . . .  1
        . x . -1
        s . .  .
        :return:  the Grid instance
        """
        g = Grid(3, 4, (2, 0))
        reward_map = {(0, 3): 1.0,
                      (1, 3): -1.0}
        action_map = {
            (0, 0): ('D', 'R'),
            (0, 1): ('L', 'R'),
            (0, 2): ('D', 'L', 'R'),
            (1, 0): ('U', 'D'),
            (1, 2): ('U', 'D', 'R'),
            (2, 0): ('U', 'R'),
            (2, 1): ('L', 'R'),
            (2, 2): ('U', 'L', 'R'),
            (2, 3): ('U', 'L'),
        }

        g.set(reward_map, action_map)

        return g

    @staticmethod
    def build_negative_grid(step_cost=-0.1):
        """
        To discourage random moves, add a small cost to each move. Penalize every move.
        :param step_cost:
        :return:
        """
        g = Grid.build_standard_grid()
        g.reward_map.update({
            (0, 0): step_cost,
            (0, 1): step_cost,
            (0, 2): step_cost,
            (1, 0): step_cost,
            (1, 2): step_cost,
            (2, 0): step_cost,
            (2, 1): step_cost,
            (2, 2): step_cost,
            (2, 3): step_cost,
        })
        return g
