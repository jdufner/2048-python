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
    # device = "cpu"
    return device


class DeepQNet(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_number, output_size) -> None:
        super().__init__()
        self.device: torch.device = determine_device()
        self.input_layer: nn.Linear = nn.Linear(input_size, hidden_size, dtype=torch.float, device=self.device)
        self.hidden_layers: list = []
        for i in range(hidden_number - 1):  # type: int
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size, dtype=torch.float, device=self.device))
        self.output_layer: nn.Linear = nn.Linear(hidden_size, output_size, dtype=torch.float, device=self.device)

    def forward(self, x) -> torch.Tensor:
        x: Tensor = F.relu(self.input_layer(x)).to(self.device)
        for hidden_layer in self.hidden_layers:  # type: nn.Linear
            x = F.relu(hidden_layer(x)).to(self.device)
        x = self.output_layer(x).to(self.device)
        return x
    
    def save(self, file_name='model.pth') -> None:
        model_folder_path: str = '../model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name: str = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def predict(self, state):
        state: np.array = np.array(state)
        state0: Tensor = torch.tensor(state, dtype=torch.float, device=self.device)
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
        state_a = np.array(state)
        state_t: Tensor = torch.tensor(state_a, dtype=torch.float, device=self.model.device)
        action: Tensor = torch.tensor(action, dtype=torch.long, device=self.model.device)
        reward: Tensor = torch.tensor(reward, dtype=torch.float, device=self.model.device)
        next_state_a = np.array(next_state)
        next_state_t: Tensor = torch.tensor(next_state_a, dtype=torch.float, device=self.model.device)

        if len(state_t.shape) == 1:
            # (1, x)
            state_t = torch.unsqueeze(state_t, 0)  # .to(self.model.device)
            action = torch.unsqueeze(action, 0)  # .to(self.model.device)
            reward = torch.unsqueeze(reward, 0)  # .to(self.model.device)
            next_state_t = torch.unsqueeze(next_state_t, 0)  # .to(self.model.device)
            done = (done, )

        # 1: predicted Q values with current state
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
