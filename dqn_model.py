import torch
import torch.nn as nn


class DQN(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)


model = DQN()

sample_state = torch.tensor(
    [-0.01, 0.02, -0.03, 0.04],
    dtype=torch.float32
)

q_values = model(sample_state)

print("Q-values:", q_values)