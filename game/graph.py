from game.agent_constants import DRAW_GRAPH
from IPython import display
import matplotlib.pyplot as plt
import numpy as np

plt.ion()


class Graph:
    plot_scores: list = []
    plot_mean_scores: list = []
    plot_moving_average: list = []
    total_score: int = 0

    @staticmethod
    def moving_average(data: list, length) -> float:
        return np.convolve(data, np.ones(length), "valid") / length

    def draw(self, score, mean_score) -> None:
        self.plot_scores.append(score)
        self.plot_mean_scores.append(mean_score)
        self.plot_moving_average.append(self.moving_average(self.plot_scores, 50))
        if DRAW_GRAPH:
            plot(self.plot_scores, self.plot_mean_scores)  # , self.plot_moving_average)


def plot(scores, mean_scores) -> None:
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
