import math
import os
import random
import time
from collections import deque
from datetime import datetime

import numpy as np

import model
from helper import plot
from model import Linear_QNet, QTrainer
from puzzle import GameGrid

MAX_MEMORY = 1_000_000
BATCH_SIZE = 10_000

LEARNING_RATE = 0.01 # alpha
GAMMA = 0.9

EPSILON_START = 0.9
EPSILON_END = 0.05
EPSILON_DECAY = 3 #_000

MAX_NUMBER_GAMES = 10 #_000

DRAW_GAME = False
DRAW_GRAPH = False

INPUT_LAYER_SIZE = 16
HIDDEN_LAYER_SIZE = 512
HIDDEN_LAYER_NUMBER = 3
OUTPUT_LAYER_SIZE = 4

class Agent:
    game_number = 0

    def __init__(self):
        self.reset()
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, HIDDEN_LAYER_NUMBER, OUTPUT_LAYER_SIZE).to(model.determine_device())
        self.trainer = QTrainer(self.model, lr=LEARNING_RATE, gamma=GAMMA)

    def get_state(self, game):
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

    def reset(self):
        self.game_number += 1
        self.number_exploration = 0
        self.number_exploitation = 0
        self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * (self.game_number) / EPSILON_DECAY)

    def get_action(self, state):
        final_move = [0, 0, 0, 0]
        # random moves: tradeoff exploration / exploitation
        # do in the beginning some random moves
        # in the first game half of the moves are random decreasing to zero after hundred games
        if random.random() < self.epsilon:
            move, move_type = self._explore_next_move()
        else:
            move, move_type = self._exploit_next_move(state)
        final_move[move] = 1
        return final_move, move_type

    def _explore_next_move(self):
        move_type = 'Exploration'
        self.number_exploration += 1
        move = random.randint(0, 3)
        return move, move_type

    def _exploit_next_move(self, state):
        move_type = 'Exploitation'
        self.number_exploitation += 1
        move = self.model.predict(state)
        return move, move_type

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

class Logger:
    graph = Graph()
    total_score = 0
    record = 0

    def __init__(self):
        self.start = time.time()
        now = datetime.now()
        model_folder_path = './log'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, f'{now: %Y-%m-%d_%Hh%Mm%Ss}.csv')
        self.file = open(file_name, 'w')

    def log_header(self):
        self.file.write(f'LEARNING_RATE {LEARNING_RATE}\n')
        self.file.write(f'GAMMA {GAMMA}\n')
        self.file.write(f'EPSILON_START {EPSILON_START}\n')
        self.file.write(f'EPSILON_END {EPSILON_END}\n')
        self.file.write(f'EPSILON_DECAY {EPSILON_DECAY}\n')
        self.file.write(f'MAX_NUMBER_GAMES {MAX_NUMBER_GAMES}\n')
        self.file.write(f'MAX_MEMORY {MAX_MEMORY}\n')
        self.file.write(f'BATCH_SIZE {BATCH_SIZE}\n')
        self.file.write(f'INPUT_LAYER_SIZE {INPUT_LAYER_SIZE}\n')
        self.file.write(f'HIDDEN_LAYER_SIZE {HIDDEN_LAYER_SIZE}\n')
        self.file.write(f'HIDDEN_LAYER_NUMBER {HIDDEN_LAYER_NUMBER}\n')
        self.file.write(f'OUTPUT_LAYER_SIZE {OUTPUT_LAYER_SIZE}\n')
        self.file.write('\n')
        self.file.write('Game;Time;Score;MeanScore;Epsilon;Exploration;Exploitation\n')

    def log_data(self, game_number, score, epsilon, exploration, exploitation):
        self.total_score += score
        if score > self.record:
            self.record = score
        now = time.time()
        t = round(now - self.start)
        mean_score = self.total_score / game_number
        print('Game', game_number, 'Time', t, 'Score', score, 'Mean score', mean_score, 'Record', self.record, 'Epsilon', epsilon, 'Exploration', exploration, 'Exploitation', exploitation)
        self.file.write(f'{game_number};{t};{score};{mean_score};{epsilon};{exploration};{exploitation}\n')
        self.graph.draw(score, mean_score)

    def close(self):
        self.file.close()

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
                #reward += math.log2(matrix[i][j]) / math.log2(max) # * factor
                reward += matrix[i][j] / max # * factor
    return reward

def reset(agent, game):
    game.reset()
    agent.reset()

def train():
    logger = Logger()
    logger.log_header()
    record = 0
    agent = Agent()
    game = GameGrid()
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

        if reward > 0:
            reward = calculate_reward(reward_matrix)

        #state_old_m = state_old.reshape(4, 4)
        #state_new_m = state_new.reshape(4, 4)
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

            logger.log_data(agent.game_number, score, agent.epsilon, agent.number_exploration, agent.number_exploitation)

            reset(agent, game)

            if agent.game_number > MAX_NUMBER_GAMES:
                break

    logger.close()

if __name__ == '__main__':
    train()
