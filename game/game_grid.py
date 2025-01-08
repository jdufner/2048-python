from datetime import datetime
import game.constants as c
import game.logic as logic
import logging
import numpy as np
import random
from tkinter import Frame, Label, CENTER


# If REINFORCEMENT_LEARNING_MODE == True then the game loop must be deactivated.
REINFORCEMENT_LEARNING_MODE: bool = True


def gen():
    return random.randint(0, c.GRID_LEN - 1)


class GameGrid(Frame):
    matrix: list
    done: bool = False
    reward_count_fields: int
    reward_sum_field: int
    reward_matrix: list
    history_matrices: list

    def __init__(self) -> None:
        Frame.__init__(self)
        self.grid()
        self.master.title('2048')
        self.master.bind("<Key>", self.key_down)
        self.commands = {
            c.KEY_UP: logic.up,
            c.KEY_DOWN: logic.down,
            c.KEY_LEFT: logic.left,
            c.KEY_RIGHT: logic.right,
            # c.KEY_BACK: self._key_down(c.KEY_BACK),
        }
        self.grid_cells = []
        self._init_grid()
        self.reset()
        self._update_grid_cells()

        # Remove comment from next line and run this file.
        if not REINFORCEMENT_LEARNING_MODE:
            self.mainloop()

    def _init_grid(self) -> None:
        background: Frame = Frame(self, bg=c.BACKGROUND_COLOR_GAME, width=c.SIZE, height=c.SIZE)
        background.grid()
        for i in range(c.GRID_LEN):  # type: int
            grid_row: list = []
            for j in range(c.GRID_LEN):  # type: int
                cell: Frame = Frame(
                    background,
                    bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                    width=c.SIZE / c.GRID_LEN,
                    height=c.SIZE / c.GRID_LEN
                )
                cell.grid(
                    row=i,
                    column=j,
                    padx=c.GRID_PADDING,
                    pady=c.GRID_PADDING
                )
                t: Label = Label(
                    master=cell,
                    text="",
                    bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                    justify=CENTER,
                    font=c.FONT,
                    width=5,
                    height=2)
                t.grid()
                grid_row.append(t)
            self.grid_cells.append(grid_row)

    def _update_grid_cells(self) -> None:
        for i in range(c.GRID_LEN):  # type: int
            for j in range(c.GRID_LEN):  # type: int
                new_number: int = self.matrix[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(text="", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(
                        text=str(new_number),
                        bg=c.BACKGROUND_COLOR_DICT[new_number],
                        fg=c.CELL_COLOR_DICT[new_number]
                    )
        self.update_idletasks()

    def key_down(self, event):
        logging.debug(f'event {event}')
        key: str = event.keysym
        self._key_down(key)

    def _key_down(self, key):
        if key == c.KEY_QUIT:
            exit()
        if key == c.KEY_BACK and len(self.history_matrices) > 1:
            self.matrix = self.history_matrices.pop()
            self._update_grid_cells()
            logging.info(f'back on step total step: {len(self.history_matrices)}')
        elif key in self.commands:
            self.reward = 0
            self.matrix, done, self.reward_count_fields, self.reward_sum_field, self.reward_matrix = (
                self.commands[key](self.matrix))
            logging.debug(f'matrix = {self.matrix}, reward_count_field = {self.reward_count_fields}, '
                          f'reward_sum_field = {self.reward_sum_field}, reward_matrix = {self.reward_matrix}')
            if self.reward_count_fields > 0:
                self.reward = self.reward_count_fields
            else:
                self.reward = -1
            if done:
                if (len(self.history_matrices) > 0 and
                        self.matrix == self.history_matrices[len(self.history_matrices) - 1]):
                    self.reward = -1
                self.matrix = logic.add_two(self.matrix)
                # record last move
                self.history_matrices.append(self.matrix)
                self._update_grid_cells()
                if logic.game_state(self.matrix) == logic.WIN:
                    self.grid_cells[1][1].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[1][2].configure(text="Win!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                    self.reward = 10_000
                    self.done = True
                if logic.game_state(self.matrix) == logic.LOSE:
                    self.grid_cells[1][1].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[1][2].configure(text="Lose!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                    self.reward = -10_000
                    self.done = True
            else:
                self.reward = -1

    def generate_next(self) -> None:
        index: tuple[int, int] = (gen(), gen())
        while self.matrix[index[0]][index[1]] != 0:
            index = (gen(), gen())
        self.matrix[index[0]][index[1]] = 2

    # Extension for Reinforcement Learning
    # Returns reward, reward_count_fields, reward_sum_field, reward_matrix, done, score
    # reward: -1 if no field(s) merged, else reward_count_field, else on loose -10,000, else on win 10,000
    # reward_count_field: the number of fields that have been merged
    # reward_sum_field: the sum of fields that have been merged
    # reward_matrix: a matrix of fields that have been merged
    # done: True if all field filled and no move possible
    # score: the sum of all fields
    #
    # Example
    #  .  .  .  .        .  .  .  .
    #  .  .  .  .  --\   .  .  .  .
    #  .  .  .  4  --/   .  .  .  .
    #  .  .  2  4        .  .  2  8
    #
    # reward: 1
    # reward_count_field: 1
    # reward_sum_field: 8
    # reward_matrix:  .  .  .  .
    #                 .  .  .  .
    #                 .  .  .  .
    #                 .  .  .  8
    # done: False
    # score: 10
    def play_step(self, action) -> tuple[int, int, int, list, bool, bool, int]:
        key: str = self._key_from_actionindex(action)
        self._key_down(key)
        score: int = logic.calculate_score(self.matrix)
        logging.debug(f'matrix = {self.matrix}, reward = {self.reward}, '
                      f'reward_count_field = {self.reward_count_fields}, '
                      f'reward_sum_field = {self.reward_sum_field}, reward_matrix = {self.reward_matrix}, '
                      f'done = {self.done}, score = {score}')
        return self.reward, self.reward_count_fields, self.reward_sum_field, self.reward_matrix, self.done, self.done, score

    # Extension for Reinforcement Learning
    @staticmethod
    def _key_from_actionarray(action) -> str:
        logging.debug(f'array {action} has max at index {np.argmax(action)}')
        if np.array_equal(action, [1, 0, 0, 0]):
            key = c.KEY_RIGHT
        elif np.array_equal(action, [0, 1, 0, 0]):
            key = c.KEY_DOWN
        elif np.array_equal(action, [0, 0, 1, 0]):
            key = c.KEY_LEFT
        else:
            key = c.KEY_UP
        return key

    @staticmethod
    def _key_from_actionindex(action) -> str:
        if 0 == action:
            key = c.KEY_RIGHT
        elif 1 == action:
            key = c.KEY_DOWN
        elif 2 == action:
            key = c.KEY_LEFT
        else:
            key = c.KEY_UP
        return key

    # Extension for Reinforcement Learning
    def reset(self) -> None:
        self.matrix = logic.new_game(c.GRID_LEN)
        self.done = False
        # self.reward = 0
        self.reward_count_fields = 0
        self.reward_sum_field = 0
        self.reward_matrix = []
        self.history_matrices = []
        self._update_grid_cells()


if __name__ == '__main__':
    now: datetime = datetime.now()
    logging.basicConfig(filename=f'../logs/{now: %Y-%m-%d_%Hh%Mm%Ss}_game.logs',
                        encoding='utf-8',
                        format='%(asctime)s,%(msecs)-3d - %(levelname)-8s - %(filename)s:%(lineno)d - '
                               '%(module)s - %(funcName)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG)
    REINFORCEMENT_LEARNING_MODE = False
    game_grid = GameGrid()
