import numpy as np
from builtins import range
import matplotlib.pyplot as plt

from grid_world.grid import Grid


def print_values(V, g: Grid):
    for i in range(g.height):
        print('------------------------')
        for j in range(g.width):
            v = V.get((i, j), 0)
            if v >= 0:
                print(' %.2f|' % v, end='')
            else:
                print('%.2f|' % v, end='')  # - sign
        print("")
    print('------------------------')


def print_policy(P, g: Grid):
    for i in range(g.height):
        print('------------------------')
        for j in range(g.width):
            a = P.get((i, j), ' ')
            print('  %s  ' % a, end='')
        print('')
    print('------------------------')
