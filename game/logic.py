#
# CS1010FC --- Programming Methodology
#
# Mission N Solutions
#
# Note that written answers are commented out to allow us to run your
# code easily while grading your problem set.

import game.constants as c
import logging
import random

#######
# Task 1a #
#######

# [Marking Scheme]
# Points to note:
# Matrix elements must be equal but not identical
# 1 mark for creating the correct matrix


def new_game(n: int) -> list:
    matrix: list = []
    for i in range(n):  # type: int
        matrix.append([0] * n)
    matrix = add_two(matrix)
    matrix = add_two(matrix)
    return matrix

###########
# Task 1b #
###########

# [Marking Scheme]
# Points to note:
# Must ensure that it is created on a zero entry
# 1 mark for creating the correct loop


def add_two(mat: list) -> list:
    a: int = random.randint(0, len(mat)-1)
    b: int = random.randint(0, len(mat)-1)
    while mat[a][b] != 0:
        a = random.randint(0, len(mat)-1)
        b = random.randint(0, len(mat)-1)
    # 2 and 4 are in an 80:20 ratio
    mat[a][b] = 2 if (random.random() < .8) else 4
    return mat

###########
# Task 1c #
###########

# [Marking Scheme]
# Points to note:
# Matrix elements must be equal but not identical
# 0 marks for completely wrong solutions
# 1 mark for getting only one condition correct
# 2 marks for getting two of the three conditions
# 3 marks for correct checking


def game_state(mat: list) -> str:
    # check for win cell
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if mat[i][j] == 2048:
                return 'win'
    # check for any zero entries
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if mat[i][j] == 0:
                return 'not over'
    # check for same cells that touch each other
    for i in range(len(mat)-1):
        # intentionally reduced to check the row on the right and below
        # more elegant to use exceptions but most likely this will be their solution
        for j in range(len(mat[0])-1):
            if mat[i][j] == mat[i+1][j] or mat[i][j+1] == mat[i][j]:
                return 'not over'
    for k in range(len(mat)-1):  # to check the left/right entries on the last row
        if mat[len(mat)-1][k] == mat[len(mat)-1][k+1]:
            return 'not over'
    for j in range(len(mat)-1):  # check up/down entries on last column
        if mat[j][len(mat)-1] == mat[j+1][len(mat)-1]:
            return 'not over'
    return 'lose'


def calculate_score(mat: list) -> int:
    score: int = 0
    for i in range(len(mat)):  # type: int
        for j in range(len(mat[0])):  # type: int
            score += mat[i][j]
    return score

###########
# Task 2a #
###########

# [Marking Scheme]
# Points to note:
# 0 marks for completely incorrect solutions
# 1 mark for solutions that show general understanding
# 2 marks for correct solutions that work for all sizes of matrices
#
# 1 2 3    3 2 1
# 4 5 6 -> 6 5 4
# 7 8 9    9 8 7


def reverse(mat: list) -> list:
    new: list = []
    for i in range(len(mat)):  # type: int
        new.append([])
        for j in range(len(mat[0])):  # type: int
            new[i].append(mat[i][len(mat[0])-j-1])
    return new

###########
# Task 2b #
###########

# [Marking Scheme]
# Points to note:
# 0 marks for completely incorrect solutions
# 1 mark for solutions that show general understanding
# 2 marks for correct solutions that work for all sizes of matrices
#
# 1 2 3    1 4 7
# 4 5 6 -> 2 5 8
# 7 8 9    3 6 9


def transpose(mat: list) -> list:
    new: list = []
    for i in range(len(mat[0])):  # type: int
        new.append([])
        for j in range(len(mat)):  # type: int
            new[i].append(mat[j][i])
    return new

##########
# Task 3 #
##########

# [Marking Scheme]
# Points to note:
# The way to do movement is compress -> merge -> compress again
# Basically if they can solve one side, and use transpose and reverse correctly they should
# be able to solve the entire thing just by flipping the matrix around
# No idea how to grade this one at the moment. I have it pegged to 8 (which gives you like,
# 2 per up/down/left/right?) But if you get one correct likely to get all correct so...
# Check the down one. Reverse/transpose if ordered wrongly will give you wrong result.


def cover_up(mat: list) -> tuple[list, bool]:
    new: list = []
    for j in range(c.GRID_LEN):  # type: int
        partial_new: list = []
        for i in range(c.GRID_LEN):  # type: int
            partial_new.append(0)
        new.append(partial_new)
    done: bool = False
    for i in range(c.GRID_LEN):  # type: int
        count: int = 0
        for j in range(c.GRID_LEN):  # type: int
            if mat[i][j] != 0:
                new[i][count] = mat[i][j]
                if j != count:
                    done: bool = True
                count += 1
    return new, done


def merge(mat: list, done: bool) -> tuple[list, bool, int, int, list]:
    reward_count_fields: int = 0
    reward_sum_field: int = 0
    reward_matrix: list = []
    for i in range(c.GRID_LEN):  # type: int
        reward_matrix.append([])
        for j in range(c.GRID_LEN-1):  # type: int
            reward_matrix[i].append(0)
            if mat[i][j] == mat[i][j+1] and mat[i][j] != 0:
                mat[i][j] *= 2
                reward_count_fields += 1
                reward_sum_field += mat[i][j]
                reward_matrix[i][j] = mat[i][j]
                mat[i][j+1] = 0
                done: bool = True
        reward_matrix[i].append(0)
    return mat, done, reward_count_fields, reward_sum_field, reward_matrix


def up(game: list) -> tuple[list, bool, int, int, list]:
    logging.debug('up')
    # return matrix after shifting up
    game = transpose(game)
    game, done = cover_up(game)
    game, done, reward_count_fields, reward_sum_field, reward_matrix = merge(game, done)
    game = cover_up(game)[0]
    game = transpose(game)
    reward_matrix = transpose(reward_matrix)
    return game, done, reward_count_fields, reward_sum_field, reward_matrix


def down(game: list) -> tuple[list, bool, int, int, list]:
    logging.debug('down')
    # return matrix after shifting down
    game = reverse(transpose(game))
    game, done = cover_up(game)
    game, done, reward_count_fields, reward_sum_field, reward_matrix = merge(game, done)
    game = cover_up(game)[0]
    game = transpose(reverse(game))
    reward_matrix = transpose(reverse(reward_matrix))
    return game, done, reward_count_fields, reward_sum_field, reward_matrix


def left(game: list) -> tuple[list, bool, int, int, list]:
    logging.debug('left')
    # return matrix after shifting left
    game, done = cover_up(game)
    game, done, reward_count_fields, reward_sum_field, reward_matrix = merge(game, done)
    game = cover_up(game)[0]
    return game, done, reward_count_fields, reward_sum_field, reward_matrix


def right(game: list) -> tuple[list, bool, int, int, list]:
    logging.debug('right')
    # return matrix after shifting right
    game = reverse(game)
    game, done = cover_up(game)
    game, done, reward_count_fields, reward_sum_field, reward_matrix = merge(game, done)
    game = cover_up(game)[0]
    game = reverse(game)
    reward_matrix = reverse(reward_matrix)
    return game, done, reward_count_fields, reward_sum_field, reward_matrix
