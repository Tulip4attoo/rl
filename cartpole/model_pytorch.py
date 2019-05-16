import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()

        # ok so the network structure is:
        # 3 layers of dense
        # in cartpole problem, state_size = 4, action_size = 2
        self.dense1 = nn.Linear(state_size, 20)
        self.dense2 = nn.Linear(20, 10)
        self.dense3 = nn.Linear(10, action_size)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        return x

