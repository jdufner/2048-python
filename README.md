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

The reward function is key to implement a reinforcement learning algorithm.

For the reward are different approaches available:

1. One point for all fields that have been merged.

    Examples:

    * 2 & 2 -> 1 point
    * 4 & 4 -> 1 point
    * 64 & 64 -> 1 point
    * 2 & 2 and 4 & 4 -> 2 points

2. The points are the sum of the added fields.

    a. The points can be added as is.

    Examples:

    * 2 & 2 -> 4 points
    * 4 & 4 -> 8 points
    * 64 & 64 -> 128 points
    * 2 & 2 and 4 & 4 -> 12 points

    b. The points can be added as log~2~.

    Examples:
    
    * 2 & 2 -> 2 points
    * 4 & 4 -> 4 points
    * 64 & 64 -> 12 points
    * 2 & 2 and 4 & 4 -> 6 points

The decision for now is to start with approach 2a.

Additionally, we could add a discount factor for the position.

|   | 0   | 1   | 2   | 3   |
|---|-----|-----|-----|-----|
| 0 | 1.0 | 0.9 | 0.9 | 1.0 |
| 1 | 0.9 | 0.8 | 0.8 | 0.9 |
| 2 | 0.9 | 0.8 | 0.8 | 0.9 |
| 3 | 1.0 | 0.9 | 0.9 | 1.0 |



| Outcome of an action                                                        | Game state |   Reward |
|-----------------------------------------------------------------------------|------------|---------:|
| No change in game state (illegal move)                                      | open       |       -1 |
| No numbers were added, but game isn't over (legal move)                     | open       |        0 |
| Two, Four or more numbers were added, but 2048 wasn't achieved (legal move) | open       |  <value> |
| No numbers were added, game over (legal move)                               | final      | -100,000 |
| Two, Four or more numbers were added, but 2048 was achieved (legal move)    | final      |  100,000 |

#### Random moves

The agent must be better than random moves.
The average score after 10,000 games ws 257.6.
This is the first goal to top.

### Player (agent)

#### Action

There are four possible actions.
Each action will be represented as a 1-dimensional array of size 4.

* Right: 0 - `[0, 1, 0, 0]`
* Down: 1 - `[0, 0, 1, 0]`
* Left: 2 - `[0, 0, 0, 1]`
* Up: 3 - `[1, 0, 0, 0]`

## Solution Strategy

### 2048 Python Game

The 2048 Python game is a clone of [yangshun/2048-python](https://github.com/yangshun/2048-python).
The game is modified to represent the environment in the reinforcement training circle.
It accepts actions from the agent and returns its state and a reward to the agent.

### Neural Network

#### Linear Network

#### Convolutional Network

### Agent

The Agent uses Deep Q Learning (DQN) from PyTorch library for training.
Maybe later I switch to advanced learning methods like PPO, AlphaZero or MuZero.

#### Play step function

The play step function is the interface provided by the game (environment) to the player (agent).

`play_step(self, action) -> state, reward, game_over`

### Learning strategy

#### Parameter

The following parameter must be adjusted to achieve the expected result.

| Parameter             | Description                                                                                                               | Range        |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------|--------------|
| Epsilon               | The randomness during the learning. At the begining it will be higher then it will decrease.                              | 0 < x < 1    |
| Gamma                 | The gamma value describes the weight of future values.                                                                    | 0 ≤ x ≤ 1    |
| Alpha (Learning Rate) | A good learning rate it 0.001. It describes the adoption rate in the neural network. For now, it will kept as a constant. | 0.0001 - 0.1 |

##### Epsilon

##### Gamma

Gamma value is a discount factor for future rewards.
It significantly influences the training process by determining the importance of future rewards.

* Gamma value of 0:
  The model considers only immediate rewards.
  It is called short-sighted or "myopic".
  This requires a very detailed reward function.

* Gamma value closer to 1:
  The model strives for long-term high rewards.

If it is equal to one, the agent values future reward JUST AS MUCH as current reward.
This means, in ten actions, if an agent does something good this is JUST AS VALUABLE as doing this action directly.
So learning doesn't work at that well at high gamma values.

Conversely, a gamma of zero will cause the agent to only value immediate rewards, which only works with very detailed reward functions.

Also - as for exploration behavior... there is actually TONS of literature on this.
All of your ideas have, 100%, been tried.
I would recommend a more detailed search, and to even start googling Decision Theory and "Policy Improvement".

##### Learning Rate

Alpha is the learning rate.
If the reward or transition function is stochastic (random), then alpha should change over time, approaching zero at infinity.
This has to do with approximating the expected outcome of a inner product (T(transition)*R(reward)), when one of the two, or both, have random behavior.

That fact is important to note.

Just adding a note on Alpha: Imagine you have a reward function that spits out 1, or zero, for a certain state action combo SA.
Now every time you execute SA, you will get 1, or 0.
If you keep alpha as 1, you will get Q-values of 1, or zero.
If it's 0.5, you will get values of +0.5, or 0, and the function will always oscillate between the two values for ever.
However, if everytime you decrease your alpha by 50 percent, you get values like this (assuming reward is recieved 1,0,1,0,...).
Your Q-values will end up being, 1,0.5,0.75,0.9,0.8,.... And will eventually converge kind of close to 0.5.
At infinity it will be 0.5, which is the expected reward in a probabilistic sense.

### Results

As described two different neural networks have been used:

1. Linear Network
2. Convolutional Network

There was no training success with the Linear Network after 10,000 games.
Therefore, I decided to move to a Convolutional Network.

#### Linear Network

TODO: list one or more training examples

#### Convolutional Network

## Installation

### Pre-requisites

Not a pre-requisite, but a recommendation is to set up a [virtual environment](https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/).

#### Python dependencies

Following libs are required:

* torch
* torchvision
* matplotlib
* ipython

#### Python lib

1. Enter your virtual environment.
2. Install libraries without `requirements.txt`.

As local admin

`pip install torch torchvision matplotlib ipython`

As Non-admin

`python -m pip install torch torchvision matplotlib ipython --index-url https://download.pytorch.org/whl/cu118`

TODO: Create a requirements.txt file to make it easier.

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
