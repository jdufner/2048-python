# 2048 Python Reinforcement Learning

## Goal

This project implements a self-trained algorithm to solve 2048 game.

## Context

![Reinforcement Training circle](img/Reinforcement%20Training%20circle.png)

| Component | Description                                                               |
|-----------|---------------------------------------------------------------------------|
| 2048 Game | In terms of reinforcement learning it is the environment.                 |
| Player    | Acts a the player. It observes and receives rewards from the environment. |

## Building Blocks

### 2048 Game (Environment)

#### State

The state will be represented as a 1-dimensional array of size 16.
It contains the fields from top to down and left to right.

Besides the state of the game, the result of the game (win, lose, not_done) will be part of the state.
Maybe the result must be part of the reward.

#### Reward

For the reward are different approaches available:

1. One point for all fields that have been merged.

    Examples:

    * 2 & 2 -> 1 point
    * 2 & 2 and 4 & 4 -> 2 points

2. The points are the sum of the added fields.

    Examples:

    * 2 & 2 -> 4 points
    * 2 & 2 and 4 & 4 -> 4 and 8 points = 12 points

The decision for now is to start with approach 2.

### Player (agent)

#### Action

There are four possible actions.
Each action will be represented as a 1-dimensional array of size 4.

* Up: `[1, 0, 0, 0]`
* Right: `[0, 1, 0, 0]`
* Down: `[0, 0, 1, 0]`
* Left: `[0, 0, 0, 1]`

## Solution Strategy

### 2048 Python Game

The 2048 Python game is a clone of [yangshun/2048-python](https://github.com/yangshun/2048-python).
The game is modified to represent the environment in the reinforcement training circle.
It accepts actions from the agent and returns its state and a reward to the agent.

#### Play step function

The play step function is the interface provided by the game (environment) to the player (agent).

`play_step(self, action) -> state, reward, game_over`

### Agent

The Agent uses Deep Q Learning (DQN) from PyTorch library for training.

### Python dependencies

Following libs are required:

* torch
* torchvision
* matplotlib
* ipython

## References

* [Reinforcement Learning : Markov-Decision Process (Part 1)](https://towardsdatascience.com/introduction-to-reinforcement-learning-markov-decision-process-44c533ebf8da)
* [Introduction to Reinforcement Learning (RL) in PyTorch](https://medium.com/analytics-vidhya/introduction-to-reinforcement-learning-rl-in-pytorch-c0862989cc0e)
* [REINFORCEMENT LEARNING (DQN) TUTORIAL](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
* [REINFORCEMENT LEARNING (PPO) WITH TORCHRL TUTORIAL](https://pytorch.org/tutorials/intermediate/reinforcement_ppo.html)
* [Introduction to RL and Deep Q Networks](https://www.tensorflow.org/agents/tutorials/0_intro_rl)

## 2048 Python

[![Run on Repl.it](https://repl.it/badge/github/yangshun/2048-python)](https://repl.it/github/yangshun/2048-python)

Based on the popular game [2048](https://github.com/gabrielecirulli/2048) by Gabriele Cirulli.
The game's objective is to slide numbered tiles on a grid to combine them to create a tile with the number 2048.
Here is a Python version that uses TKinter!

![screenshot](img/screenshot.png)

To start the game, run:

    $ python3 puzzle.py

## Contributors

- [Yanghun Tay](http://github.com/yangshun)
- [Emmanuel Goh](http://github.com/emman27)
