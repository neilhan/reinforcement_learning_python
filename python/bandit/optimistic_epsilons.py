from __future__ import print_function, absolute_import, division

import numpy as np
import matplotlib.pyplot as plt
from bandit.comparing_epsilons import run_experiment as run_experiment_eps

xrange = range

class Bandit:
    def __init__(self, m, opt_mean):
        """
        :param m:  mean of the Bandit
        """
        self.m = m
        self.mean = opt_mean
        self.N = 1

    def pull(self):
        return np.random.randn() + self.m

    def update(self, x):
        self.N += 1
        self.mean = (1 - 1.0 / self.N) * self.mean + 1.0 / self.N * x


def run_experiment(m1, m2, m3, N, upper_mean=10.0):
    # set upper optimistic_mean and the mean
    bandits = [Bandit(m1, upper_mean),
               Bandit(m2, upper_mean),
               Bandit(m3, upper_mean)]

    data = np.empty(N)

    for i in xrange(N):
        # optimistic initial values has been set
        j = np.argmax([b.mean for b in bandits])

        x = bandits[j].pull()
        bandits[j].update(x)

        # for the plot
        data[i] = x

    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)

    # plot moving average ctr
    plt.plot(cumulative_average)
    plt.plot(np.ones(N) * m1)
    plt.plot(np.ones(N) * m2)
    plt.plot(np.ones(N) * m3)
    plt.xscale('log')
    plt.show()

    for i, b in enumerate(bandits):
        print(i, b.mean)

    return cumulative_average


def main():
    c_1 = run_experiment_eps(1.0, 2.0, 3.0, 0.1, 100000)
    oiv = run_experiment(1.0, 2.0, 3.0, 100000)

    # log scale plot
    plt.plot(c_1, label='eps=0.1')
    plt.plot(oiv, label='optimistic')
    plt.legend()
    plt.xscale('log')
    plt.show()


if __name__ == '__main__':
    main()
