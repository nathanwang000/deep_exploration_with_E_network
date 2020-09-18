import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

class BridgeDQN(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()
        
        self.regressor = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        x = self.regressor(x)
        return x

class PacwomanDQN(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2), # 210 x 160
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2), # 103 x 73
            nn.BatchNorm2d(32),
            nn.ReLU(),            
            nn.Conv2d(32, 32, kernel_size=5, stride=2), # 49 x 34
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.regressor = nn.Sequential(
            nn.Linear(23 * 17 * 32, 9)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        return x

class MountainCarDQN(nn.Module):

    def __init__(self):
        super(self.__class__, self).__init__()
        
        self.regressor = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.regressor(x)
        return x
    
class CartPoleDQN(nn.Module):

    def __init__(self):
        super(self.__class__, self).__init__()

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

def selectNet(netname, envname):
    DQN = {'cart_pole':  CartPoleDQN, 'pacwoman': PacwomanDQN,
           'mountain_car': MountainCarDQN,'mountain_car_long':MountainCarDQN, 'bridge': BridgeDQN}[envname]
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
