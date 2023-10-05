import torch
import random
import numpy as np
from collections import deque
from puzzle import GameGrid
from model import Linear_QNet
from model import QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1_000
LR = 0.001

class Agent:

    def __init__(self):
        self.number_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate 0 <= gamma < 1
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(16, 256, 4)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        return np.asanyarray(game.matrix).ravel()

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

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        # do in the beginning some random moves
        # in the first game half of the moves are random decreasing to zero after hundred games
        final_move = [0, 0, 0, 0]
        self.epsilon = 1_000 - self.number_games
        if random.randint(0, 2_000) < self.epsilon:
            move_type = 'Exploration'
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            move_type = 'Exploitation'
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move, move_type

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = GameGrid()
    while True:
        # get old state
        state_old = agent.get_state(game)
        game.update()

        # get move
        final_move, move_type = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        #print ('State_old', state_old, 'Move', final_move, 'Move_type', move_type, 'State_new', state_new, 'Reward', reward, 'Score', score, 'Done', done)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.number_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                # agent.model.save()

            print('Game', agent.number_games, 'Score', score, 'Record', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.number_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

        if agent.number_games >= 1:
            break

if __name__ == '__main__':
    train()
