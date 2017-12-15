import torch
import matplotlib
import os
from collections import defaultdict

EsCounter = defaultdict(list)

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

def set_log_path(log_path):
    os.system('mkdir -p %s' % log_path)
    return log_path

# environment specific settings
class DefaultSetting:
    def __init__(self):
        self.batch_size = 128
        self.lr = 0.01
        self.memory_size = 10000
        self.num_episodes = 30
        
        self.target_update_frequency_Q = 20
        self.target_update_frequency_E = 20
        self.qnet_update_frequency = 1
        self.enet_update_frequency = 1
        self.gamma_q = 0.99
        self.gamma_e = 0.99

class PacwomanSetting:
    def __init__(self):
        self.batch_size = 32
        self.lr = 0.00025
        self.memory_size = 50000
        self.num_episodes = 3000
        
        self.target_update_frequency_Q = 10000
        self.target_update_frequency_E = 10000
        self.qnet_update_frequency = 4
        self.enet_update_frequency = 4
        self.gamma_q = 0.99
        self.gamma_e = 0.99
        
class BridgeSetting:
    def __init__(self):
        self.batch_size = 30
        self.lr = 0.01
        self.memory_size = 10000
        self.num_episodes = 1000
        
        self.target_update_frequency_Q = 20
        self.target_update_frequency_E = 20
        self.qnet_update_frequency = 1
        self.enet_update_frequency = 1
        self.gamma_q = 0.99
        self.gamma_e = 0.99
        
# class MountainCarSetting(DefaultSetting):
#     def __init__(self):
#         super(self.__class__, self).__init__()
        
#         self.batch_size = 32
#         self.lr = 1e-4
#         self.memory_size = 50000        
#         self.num_episodes = 3000        

#         self.target_update_frequency_Q = 10000
#         self.target_update_frequency_E = 1000
#         self.qnet_update_frequency = 4
#         self.enet_update_frequency = 4        
#         self.gamma_q = 0.99
#         self.gamma_e = 0.99 # 2 experiments to run here: 0.99, 0.9

class MountainCarSetting(DefaultSetting):
    def __init__(self):
        super(self.__class__, self).__init__()
        
        self.batch_size = 32
        self.lr = 1e-3
        self.memory_size = 50000        
        self.num_episodes = 3000        

        self.target_update_frequency_Q = 500
        self.target_update_frequency_E = 500
        self.qnet_update_frequency = 1
        self.enet_update_frequency = 1        
        self.gamma_q = 1
        self.gamma_e = 1 # 2 experiments to run here: 0.99, 0.9
        

def getSetting(game):
    if game == 'mountain_car':
        return MountainCarSetting()
    elif game == 'bridge':
        return BridgeSetting()
    elif game == 'pacwoman':
        return PacwomanSetting()
    else:
        return DefaultSetting()
    
