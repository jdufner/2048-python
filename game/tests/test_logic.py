import game.logic as logic
import logging
import unittest


class LogicTestCase(unittest.TestCase):

    def testReverse(self):
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        logging.debug(matrix)
        reversed_matrix = logic._reverse(matrix)
        logging.debug(reversed_matrix)
        self.assertEqual(matrix[0][0], reversed_matrix[0][2])
        self.assertEqual(matrix[0][1], reversed_matrix[0][1])
        self.assertEqual(matrix[0][2], reversed_matrix[0][0])
        self.assertEqual(matrix[1][0], reversed_matrix[1][2])
        self.assertEqual(matrix[1][1], reversed_matrix[1][1])
        self.assertEqual(matrix[1][2], reversed_matrix[1][0])
        self.assertEqual(matrix[2][0], reversed_matrix[2][2])
        self.assertEqual(matrix[2][1], reversed_matrix[2][1])
        self.assertEqual(matrix[2][2], reversed_matrix[2][0])

    def testTranspose(self):
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        logging.debug(matrix)
        transposed_matrix = logic._transpose(matrix)
        logging.debug(transposed_matrix)
        self.assertEqual(matrix[0][0], transposed_matrix[0][0])
        self.assertEqual(matrix[0][1], transposed_matrix[1][0])
        self.assertEqual(matrix[0][2], transposed_matrix[2][0])
        self.assertEqual(matrix[1][0], transposed_matrix[0][1])
        self.assertEqual(matrix[1][1], transposed_matrix[1][1])
        self.assertEqual(matrix[1][2], transposed_matrix[2][1])
        self.assertEqual(matrix[2][0], transposed_matrix[0][2])
        self.assertEqual(matrix[2][1], transposed_matrix[1][2])
        self.assertEqual(matrix[2][2], transposed_matrix[2][2])

    # play_step
    def test_given_matrix_when_right_then_no_merge_expected_negative_reward(self):
        # arrange
        g: logic.Game = logic.Game()
        matrix_in = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2], [0, 0, 0, 2]]
        g.matrix = matrix_in

        # act
        reward, reward_count_fields, reward_sum_field, reward_matrix, reward_fields, terminated, truncated, score = (
            g.play_step(logic.ACTION_RIGHT))
        matrix_out = g.matrix

        # assert
        self.assertEqual(matrix_in, matrix_out)
        self.assertEqual(-1, reward)
        self.assertEqual(0, reward_count_fields)
        self.assertEqual(0, reward_sum_field)
        self.assertEqual([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], reward_matrix)
        self.assertEqual([], reward_fields)
        self.assertEqual(False, terminated)
        self.assertEqual(False, truncated)
        self.assertEqual(4, score)

    def test_given_matrix_when_right_then_no_merge_expected_zero_reward(self):
        # arrange
        g: logic.Game = logic.Game()
        matrix_in = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0], [0, 0, 0, 2]]
        g.matrix = matrix_in

        # act
        reward, reward_count_fields, reward_sum_field, reward_matrix, reward_fields, terminated, truncated, score = (
            g.play_step(logic.ACTION_RIGHT))

        # assert
        self.assertEqual(0, reward)
        self.assertEqual(0, reward_count_fields)
        self.assertEqual(0, reward_sum_field)
        self.assertEqual([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], reward_matrix)
        self.assertEqual([], reward_fields)
        self.assertEqual(False, terminated)
        self.assertEqual(False, truncated)
        self.assertEqual(4, score)

    def test_given_matrix_when_down_then_merge_expected_reward_4(self):
        # arrange
        g: logic.Game = logic.Game()
        matrix_in = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2], [0, 0, 0, 2]]
        g.matrix = matrix_in

        # act
        reward, reward_count_fields, reward_sum_field, reward_matrix, reward_fields, terminated, truncated, score = (
            g.play_step(logic.ACTION_DOWN))

        # assert
        self.assertEqual(4, reward)
        self.assertEqual(1, reward_count_fields)
        self.assertEqual(4, reward_sum_field)
        self.assertEqual([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 4]], reward_matrix)
        self.assertEqual([4], reward_fields)
        self.assertEqual(False, terminated)
        self.assertEqual(False, truncated)
        self.assertEqual(4, score)

    def test_given_matrix_when_down_then_merge_expected_reward_12(self):
        # arrange
        g: logic.Game = logic.Game()
        matrix_in = [[4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2], [4, 0, 0, 2]]
        g.matrix = matrix_in

        # act
        reward, reward_count_fields, reward_sum_field, reward_matrix, reward_fields, terminated, truncated, score = (
            g.play_step(logic.ACTION_DOWN))

        # assert
        self.assertEqual(12, reward)
        self.assertEqual(2, reward_count_fields)
        self.assertEqual(12, reward_sum_field)
        self.assertEqual([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [8, 0, 0, 4]], reward_matrix)
        self.assertEqual([8, 4], reward_fields)
        self.assertEqual(False, terminated)
        self.assertEqual(False, truncated)
        self.assertEqual(12, score)

    def test_given_matrix_when_down_then_merge_expect_win(self):
        # arrange
        g: logic.Game = logic.Game()
        matrix_in = [[2, 4, 8, 4], [4, 2, 4, 8], [2, 4, 8, 1024], [4, 8, 2, 1024]]
        g.matrix = matrix_in

        # act
        reward, reward_count_fields, reward_sum_field, reward_matrix, reward_fields, terminated, truncated, score = (
            g.play_step(logic.ACTION_DOWN))

        # assert
        self.assertEqual(100_000, reward)
        self.assertEqual(1, reward_count_fields)
        self.assertEqual(2048, reward_sum_field)
        self.assertEqual([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2048]], reward_matrix)
        self.assertEqual([2048], reward_fields)
        self.assertEqual(True, terminated)
        self.assertEqual(False, truncated)
        self.assertEqual(2112, score)

    def test_given_matrix_when_down_then_merge_expect_lose(self):
        # arrange
        g: logic.Game = logic.Game()
        matrix_in = [[2, 4, 8, 4], [4, 2, 4, 8], [2, 4, 8, 4], [4, 8, 2, 8]]
        g.matrix = matrix_in

        # act
        reward, reward_count_fields, reward_sum_field, reward_matrix, reward_fields, terminated, truncated, score = (
            g.play_step(logic.ACTION_DOWN))

        # assert
        self.assertEqual(-100_000, reward)
        self.assertEqual(0, reward_count_fields)
        self.assertEqual(0, reward_sum_field)
        self.assertEqual([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], reward_matrix)
        self.assertEqual([], reward_fields)
        self.assertEqual(False, terminated)
        self.assertEqual(True, truncated)
        self.assertEqual(76, score)


if __name__ == '__main__':
    unittest.main()
