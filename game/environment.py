import random

import numpy as np

from game.game_grid import GameGrid
from game.logic import Game
import game.constants as c


class Actions:
    actions = [c.KEY_RIGHT, c.KEY_DOWN, c.KEY_LEFT, c.KEY_UP]
    n = len(actions)

    def sample(self) -> np.array:
        # return random.randint(0, len(self.actions) - 1)
        return random.randrange(len(self.actions))


class Environment:
    visual: bool = False
    isInitialized: bool = False
    game: (GameGrid | Game)

    def reset(self) -> (list, bool):
        if self.visual:
            # Option 1
            # game.destroy() is need to rest Tk window
            if self.isInitialized:
                self.game.destroy()
            self.game = GameGrid()
            self.isInitialized = True
        else:
            # Option 2
            self.game = Game()
            self.game.reset()
            return self.flatten(self.game.matrix), False

    def step(self, action) -> (list, ):
        reward, reward_count_fields, reward_sum_field, reward_matrix, terminated, truncated, score = self.game.play_step(action)
        # returns observation, reward, terminated, truncated, info
        return self.flatten(self.game.matrix), reward_sum_field, terminated, truncated, False

    @property
    def action_space(self) -> Actions:
        return Actions()

    @staticmethod
    def flatten(xss):
        return [x for xs in xss for x in xs]
