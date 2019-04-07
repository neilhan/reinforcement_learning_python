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

    PYTHONPATH=./python:$PYTHONPATH python tic_tac_toe/play.py
