from collections import deque
from datetime import datetime
from rl.agent_constants import *
import rl.model as model
from rl.model import DeepQNet, QTrainer
from rl.my_logger import MyLogger
from game.game_grid import GameGrid
import logging
import math
import numpy as np
import random


class Agent:
    game_number: int = 0
    number_exploration: int = 0
    number_exploitation: int = 0
    epsilon: float = 0.9

    def __init__(self) -> None:
        self.reset()
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model: DeepQNet = (DeepQNet(INPUT_LAYER_SIZE, OUTPUT_LAYER_SIZE)
                                .to(model.determine_device()))
        self.trainer: QTrainer = QTrainer(self.model, lr=LEARNING_RATE, gamma=GAMMA)

    @staticmethod
    def get_state(game) -> list:
        # Better approach
        # state: np.array = np.array(np.ravel(game.matrix), dtype=int)
        # return a 2d matrix
        return game.matrix
        # returns a flattened game.matrix
        # return np.array(np.ravel(game.matrix), dtype=int)

    def remember(self, state, action, reward, next_state, done) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self) -> None:
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done) -> None:
        self.trainer.train_step(state, action, reward, next_state, done)

    def reset(self):
        self.game_number += 1
        self.number_exploration = 0
        self.number_exploitation = 0
        self.epsilon: float = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * self.game_number /
                                                                                     EPSILON_DECAY)

    def get_action(self, state: list) -> tuple[list[int], str]:
        final_move: list = [0, 0, 0, 0]
        # random moves: tradeoff exploration / exploitation
        # do in the beginning some random moves
        # in the first game half of the moves are random decreasing to zero after hundred games
        if random.random() < self.epsilon:
            move, move_type = self._explore_next_move()
        else:
            move, move_type = self._exploit_next_move(state)
        final_move[move] = 1
        return final_move, move_type

    def _explore_next_move(self) -> tuple[int, str]:
        move_type: str = 'Exploration'
        self.number_exploration += 1
        move = random.randint(0, 3)
        return move, move_type

    def _exploit_next_move(self, state: list) -> tuple[int, str]:
        move_type: str = 'Exploitation'
        self.number_exploitation += 1
        move: int = self.model.predict(state)
        return move, move_type


def calculate_reward(matrix) -> float:
    reward: float = 0.0
    max_value: int = np.max(matrix)
    for i in range(len(matrix)):  # type: int
        for j in range(len(matrix[i])):  # type: int
            factor: float = 1.0
            if (i == 0 or i == len(matrix) - 1) and (j == 0 or j == len(matrix[i]) - 1):
                factor *= 1.1
            if (i != 0 and i != len(matrix) - 1) and (j != 0 and j != len(matrix[i]) - 1):
                factor *= 0.9
            if matrix[i][j] > 0:
                # reward += math.log2(matrix[i][j]) / math.log2(max) # * factor
                reward += matrix[i][j] / max_value  # * factor
    return reward


def reset(agent, game) -> None:
    game.reset()
    agent.reset()


def train() -> None:
    my_logger: MyLogger = MyLogger()
    my_logger.log_header()
    agent: Agent = Agent()
    game: GameGrid = GameGrid()
    record: int = 0
    epoch: int = 0
    while True:
        epoch += 1
        # get old state
        state_old: list = agent.get_state(game)
        logging.debug(f'game: {agent.game_number}, epoch: {epoch}, state_old: {game.matrix}')
        if DRAW_GAME:
            game.update()
        # get move
        final_move, move_type = agent.get_action(state_old)
        logging.debug(f'game: {agent.game_number}, epoch: {epoch}, move: {final_move}, type: {move_type}')
        # perform move and get new state
        reward, reward_count_fields, reward_sum_field, reward_matrix, done, score = game.play_step(final_move)  # type:
        # [int, int, int, list, bool, int]
        # if reward > 0:
        #     reward = calculate_reward(reward_matrix)
        state_new: list = agent.get_state(game)
        logging.debug(f'game: {agent.game_number}, epoch: {epoch}, state_new: {game.matrix}')
        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        # remember
        agent.remember(state_old, final_move, reward, state_new, done)
        if done:
            # train long memory, plot result
            agent.train_long_memory()
            if score > record:
                record = score
                # Why save() without load()?
                # agent.model.save()
            my_logger.log_data(agent.game_number, score, agent.epsilon, agent.number_exploration,
                               agent.number_exploitation)
            reset(agent, game)
            if agent.game_number > MAX_NUMBER_GAMES:
                break
    my_logger.close()


if __name__ == '__main__':
    now: datetime = datetime.now()
    logging.basicConfig(filename=f'./logs/{now: %Y-%m-%d_%Hh%Mm%Ss}_agent.log',
                        encoding='utf-8',
                        format='%(asctime)s,%(msecs)-3d - %(levelname)-8s - %(filename)s:%(lineno)d - '
                               '%(module)s - %(funcName)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
    train()
