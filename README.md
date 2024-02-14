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
    * 2 & 2 and 4 & 4 -> 2 points

2. The points are the sum of the added fields.

    Examples:

    * 2 & 2 -> 4 points
    * 4 & 4 -> 8 points
    * 2 & 2 and 4 & 4 -> 12 points

3. The points are independent of the added fields.

    Examples:

    * 2 & 2 -> 10 points
    * 4 & 4 -> 10 points
    * 2 & 2 and 4 & 4 -> 10 points

4. The points depend on the score and the number of tiles.

    Examples

    * 2x 2 on the board -> Score = 4, Tiles = 2 -> 4/2 = 2 points
    * 1x 2 and 1x 4 on the board -> Score = 6, Tiles = 2 -> 6/2 = 3 points
    * 1x 2, 1x 4 and 1x 8 on the board -> Score = 14, Tiles = 3 -> 14/3 = 4,7 points

The decision for now is to start with approach 2.

| Outcome of an action                                                        | Game state | Reward |
|-----------------------------------------------------------------------------|------------|-------:|
| No change in game state (illegal move)                                      | open       |     -1 |
| No numbers were added, but game isn't over (legal move)                     | open       |    -10 |
| Two, Four or more numbers were added, but 2048 wasn't achieved (legal move) | open       |     10 |
| No numbers were added, game over (legal move)                               | final      |  -1000 |
| Two, Four or more numbers were added, but 2048 was achieved (legal move)    | final      |   1000 |

#### Random moves

The agent must be better than random moves.
The average score after 10,000 games ws 257.6.
This is the first goal to top.

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

### Neural Network

How do I need to design my neural network to represent the problem best?

#### Linear Network

Bard / Gemini:

> Linear network:
>
> While the underlying state of the 2048 game can be represented as a 16x16 matrix, processing it directly with a linear network wouldn't capture the spatial relationships between tiles, which are crucial for making optimal moves.
> Linear networks are designed for processing vectorized data, not grid-like structures like images.

#### Convolutional Networ

Bard / Gemini:

> Convolutional network:
>
> Although CNNs excel at analyzing grid-like data, the 2048 game doesn't require complex feature extraction like edge detection or texture analysis, which are CNN's strengths.
> Using a full-fledged CNN for such a simple game might be overkill and computationally expensive.

### Reinforcement Learning Algorithm

#### DQN

#### PPO

#### Play step function

The play step function is the interface provided by the game (environment) to the player (agent).

`play_step(self, action) -> state, reward, game_over`

### Agent

The Agent uses Deep Q Learning (DQN) from PyTorch library for training.

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
  The model strives for long-termin high rewards.

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

## Installation

### Pre-requisites

Not a pre-requisite, but a recommendation is to set up a virtual environment, see [python.org](https://docs.python.org/3/library/venv.html) or [freecodecamp.org](https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/).

`python -m venv .venv`

`. ./venv/Scripts/activate`

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

`python -m pip install torch torchvision matplotlib ipython`

Create a requirements.txt file to make it easier.

`pip freeze > requirements.txt`

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
