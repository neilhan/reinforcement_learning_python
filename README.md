# RL practise

Course: [https://www.udemy.com/artificial-intelligence-reinforcement-learning-in-python/ ](https://www.udemy.com/artificial-intelligence-reinforcement-learning-in-python/)

## A docker image for python3 and lib
The code runs with this docker setup: [https://github.com/neilhan/docker_collection/tree/master/dnn_gui_py3 ](https://github.com/neilhan/docker_collection/tree/master/dnn_gui_py3)

    ./build.sh
    ./run.sh

You will be given a shell. Expecting ~/Projects/reinforcement_learning_python.

## python/tic_tac_toe

The naive solution to Tic tac toe game. Value function updated after every game play.

## to run any code

```sh

O
PYTHONPATH=./python:$PYTHONPATH python3 python/tic_tac_toe/play.py
# or
PYTHONPATH=./python:$PYTHONPATH python3 tic_tac_toe.play

PYTHONPATH=./python python3 python/grid_world/iterative_policy_evaluation.py
# or
PYTHONPATH=./python python3 -m grid_world.iterative_policy_evaluation
```

## Files
- tic_tac_toe/play.py - into to naive solution
- Bandit
  - ucb1.py - Upper confidence bound
  - comparing_epsilons.py
  - comparing_explore_exploit_imethods.py - BayesianBandit
  - optimistic_epsilons.py - optimistic initial value
- grid_world
  - iterative_policy_evaluation.py - intro to policy evaluation
  - policy_iteration.py - policy eval + policy improvement
