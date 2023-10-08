import torch
import random
import numpy as np
from collections import deque
from puzzle import GameGrid
from model import Linear_QNet
from model import QTrainer
from helper import plot
import math
import time

MAX_MEMORY = 100_000
BATCH_SIZE = 1_000

LEARNING_RATE = 0.001 # alpha
GAMMA = 0.9

EPSILON_START = 0.9
EPSILON_END = 0.05
EPSILON_DECAY = 3_000

MAX_NUMBER_GAMES = 20_000

DRAW_GAME = False
DRAW_GRAPH = False

class Agent:

    def __init__(self):
        self.number_games = 1
        self.epsilon = EPSILON_START # randomness
        self.number_exploration = 0
        self.number_exploitation = 0
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(16, 512, 4)
        self.trainer = QTrainer(self.model, lr=LEARNING_RATE, gamma=GAMMA)

    def get_state(self, game):
        #return np.asanyarray(game.matrix).ravel()
        return np.array(np.ravel(game.matrix), dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def calculate_epsilon(self):
        # Old exploration calculation
        #self.epsilon = 10_000 - self.number_games
        #if random.randint(0, 40_000) < self.epsilon:
        # New exploration calculation
        self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * self.number_games / EPSILON_DECAY)

    def get_action(self, state):
        final_move = [0, 0, 0, 0]
        # random moves: tradeoff exploration / exploitation
        # do in the beginning some random moves
        # in the first game half of the moves are random decreasing to zero after hundred games
        if random.random() < self.epsilon:
            move_type = 'Exploration'
            self.number_exploration += 1
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            move_type = 'Exploitation'
            self.number_exploitation += 1
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move, move_type

class Graph:
    plot_scores = []
    plot_mean_scores = []
    plot_moving_average = []
    total_score = 0

    def moving_average(self, data, length):
        return np.convolve(data, np.ones(length), "valid") / length

    def draw(self, score, mean_score):
        self.plot_scores.append(score)
        self.plot_mean_scores.append(mean_score)
        self.plot_moving_average.append(self.moving_average(self.plot_scores, 50))
        if DRAW_GRAPH:
            plot(self.plot_scores, self.plot_mean_scores) #, self.plot_moving_average)

def calculate_reward(matrix):
    reward = 0.0
    max = np.max(matrix)
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            factor = 1.0
            if (i == 0 or i == len(matrix) - 1) and (j == 0 or j == len(matrix[i]) - 1):
                factor *= 1.1
            if (i != 0 and i != len(matrix) - 1) and (j != 0 and j != len(matrix[i]) - 1):
                factor *= 0.9
            if matrix[i][j] > 0:
                reward += math.log2(matrix[i][j]) / math.log2(max) # * factor
    return reward

def train():
    graph = Graph()
    total_score = 0
    record = 0
    agent = Agent()
    game = GameGrid()
    agent.calculate_epsilon()
    start = time.time()
    while True:
        # get old state
        state_old = agent.get_state(game)
        if DRAW_GAME:
            game.update()

        # get move
        final_move, move_type = agent.get_action(state_old)

        # perform move and get new state
        reward, reward_count_fields, reward_sum_field, reward_matrix, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        state_old_m = state_old.reshape(4, 4)
        state_new_m = state_new.reshape(4, 4)

        if reward > 0:
            #reward = reward_sum_field / np.max(reward_matrix)
            reward = calculate_reward(reward_matrix)

        #print ('State_old\n', state_old_m, '\nState_new\n', state_new_m, 'Move', game.key_from_action(final_move), 'Move_type', move_type, 'Reward', reward, 'Score', score, 'Done', done)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            # Plot
            total_score += score
            mean_score = total_score / agent.number_games
            graph.draw(score, mean_score)

            now = time.time()

            print('Game', agent.number_games, 'Time', round(now - start) , 'Score', score, 'Mean score', mean_score, 'Record', record, 'Epsilon', agent.epsilon, 'Exploration', agent.number_exploration, 'Exploitation', agent.number_exploitation)

            game.reset()
            agent.number_games += 1
            agent.number_exploration = 0
            agent.number_exploitation = 0
            agent.calculate_epsilon()
        if agent.number_games > MAX_NUMBER_GAMES:
            break

if __name__ == '__main__':
    train()
