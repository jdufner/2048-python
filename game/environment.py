import random

import numpy as np

from game.game_grid import GameGrid
import game.constants as c


class Actions:
    actions = [c.KEY_RIGHT, c.KEY_DOWN, c.KEY_LEFT, c.KEY_UP]
    n = len(actions)

    def sample(self) -> np.array:
        action: list = [0, 0, 0, 0]
        index = random.randint(0, 3)
        action[index] = 1
        # return np.array(action)
        return index


class Environment:
    game: GameGrid

    def reset(self) -> (list, bool):
        self.game = GameGrid()
        # np.array(np.ravel(game.matrix), dtype=int)
        return self.flatten(self.game.matrix), False

    def step(self, action):

        reward, reward_count_fields, reward_sum_field, reward_matrix, done, score = self.game.play_step(action)
        return self.flatten(self.game.matrix), reward, done, False, False

    @property
    def action_space(self) -> Actions:
        return Actions()

    @staticmethod
    def flatten(xss):
        return [x for xs in xss for x in xs]