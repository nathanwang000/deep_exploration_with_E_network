import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),            
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.regressor = nn.Sequential(
            nn.Linear(448, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        return x

def selectNet(netname='dqn'):
    if netname == 'dqn':
        return DQN()
    elif netname == 'enet':  # optimistic start and dora
        # zero initialize last layer, sigmoid activation at the end
        Enet = DQN()
        for p in Enet.regressor[-1].parameters():
            p.data.fill_(0)
        Enet.regressor = nn.Sequential(*(list(Enet.regressor.children()) + \
                                         [nn.Sigmoid()]))
        return Enet
    else: 
        raise Exception('not implemented error')
