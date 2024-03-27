from datetime import datetime
from rl.agent_constants import *
from rl.graph import Graph
import io
import os
import time


class MyLogger:
    graph: Graph = Graph()
    total_score: int = 0
    record: int = 0

    def __init__(self) -> None:
        self.start: time = time.time()
        now: datetime = datetime.now()
        model_folder_path: str = './logs'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name: str = os.path.join(model_folder_path, f'{now: %Y-%m-%d_%Hh%Mm%Ss}.csv')
        self.file: io.open = open(file_name, 'w')

    def log_header(self) -> None:
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

    def log_data(self, game_number: int, score: int, epsilon: float, exploration: int, exploitation: int) -> None:
        self.total_score += score
        if score > self.record:
            self.record = score
        now: time = time.time()
        t: int = round(now - self.start)
        mean_score: float = self.total_score / game_number
        print('Game', game_number, 'Time', t, 'Score', score, 'Mean score', mean_score, 'Record', self.record,
              'Epsilon', epsilon, 'Exploration', exploration, 'Exploitation', exploitation)
        self.file.write(f'{game_number};{t};{score};{mean_score};{epsilon};{exploration};{exploitation}\n')
        self.graph.draw(score, mean_score)

    def close(self) -> None:
        self.file.close()
