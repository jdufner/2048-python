import logging

import numpy as np
import os
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss
import torch.optim as optim
from torch.optim import Optimizer


def determine_device() -> torch.device:
    # if GPU is to be used
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


class DeepQNet(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super().__init__()
        self.device: torch.device = determine_device()
        self.input_layer: nn.Conv2d = nn.Conv2d(1, 10, 2, 1, 1, dtype=torch.float, device=self.device)
        self.act = nn.ReLU()
        self.flat = nn.Flatten(start_dim=0, end_dim=-1)
        self.output_layer: nn.Linear = nn.Linear(250, output_size, dtype=torch.float, device=self.device)

    def forward(self, x) -> torch.Tensor:
        x: Tensor = self.input_layer(x).to(self.device)
        logging.info(f'conv2d = {x.size()}')
        x: Tensor = self.act(x).to(self.device)
        logging.info(f'relu = {x.size()}')
        x: Tensor = self.flat(x).to(self.device)
        logging.info(f'flat = {x.size()}')
        x = self.output_layer(x).to(self.device)
        return x

    def save(self, file_name='model.pth') -> None:
        model_folder_path: str = '../model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name: str = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def predict(self, state):
        # state: np.array = np.array(state)
        state0: Tensor = torch.tensor(state, dtype=torch.float, device=self.device)
        state0 = torch.unsqueeze(state0, 0)
        prediction: Tensor = self(state0)
        move: int = torch.argmax(prediction).item()
        return move


class QTrainer:
    def __init__(self, model: DeepQNet, lr: float, gamma: float) -> None:
        self.model: DeepQNet = model
        self.lr: float = lr
        self.gamma: float = gamma
        self.optimizer: Optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion: MSELoss = nn.MSELoss().to(self.model.device)

    def train_step(self, state: list, action: Tensor, reward: float, next_state: list, done: bool) -> None:
        # state_a = np.array(state)
        state_t: Tensor = torch.tensor(state, dtype=torch.float, device=self.model.device)
        action: Tensor = torch.tensor(action, dtype=torch.long, device=self.model.device)
        reward: Tensor = torch.tensor(reward, dtype=torch.float, device=self.model.device)
        # next_state_a = np.array(next_state)
        next_state_t: Tensor = torch.tensor(next_state, dtype=torch.float, device=self.model.device)

        # State must be a 1-dimensional tensor
        # if len(state_t.shape) == 1:
        #     # (1, x)
        #     state_t = torch.unsqueeze(state_t, 0)  # .to(self.model.device)
        #     action = torch.unsqueeze(action, 0)  # .to(self.model.device)
        #     reward = torch.unsqueeze(reward, 0)  # .to(self.model.device)
        #     next_state_t = torch.unsqueeze(next_state_t, 0)  # .to(self.model.device)
        #     done = (done,)
        # else:
        #     logging.debug(f'state_t.shape = {state_t.shape}')

        # 1: predicted Q values with current state
        state_t = torch.unsqueeze(state_t, 0)
        prediction: Tensor = self.model(state_t)
        target: Tensor = prediction.clone()
        for idx in range(len(done)):  # type: int
            Q_new: Tensor = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state_t[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 2: Q_new = r + y * max(next_predicted Q value)
        # prediction.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss: MSELoss = self.criterion(target, prediction)
        loss.backward()
        self.optimizer.step()
