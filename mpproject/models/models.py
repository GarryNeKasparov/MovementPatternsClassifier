import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropouts: list,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropouts[0],
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn = nn.BatchNorm1d(hidden_size // 2)
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(p=dropouts[1])
        self.fc2 = nn.Linear(hidden_size // 2, 2)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        x, _ = self.lstm(x, (h_0, c_0))
        x = x[:, -1]
        x = self.fc1(x)
        x = self.bn(x)
        x = self.dp(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class GRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropouts: list,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropouts[0],
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn = nn.BatchNorm1d(hidden_size // 2)
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(p=dropouts[1])
        self.fc2 = nn.Linear(hidden_size // 2, 2)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        x, _ = self.gru(x, h_0)
        x = x[:, -1]
        x = self.fc1(x)
        x = self.bn(x)
        x = self.dp(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
