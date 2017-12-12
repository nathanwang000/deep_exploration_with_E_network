import torch
import matplotlib
import os

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

working_dir= os.getcwd()
log_path = os.path.join(working_dir, 'logs/')
os.system('mkdir -p %s' % log_path)

# environment specific settings
def mountain_car_setting():
    batch_size = 32
    target_update_frequency_Q = 10000
    target_update_frequency_E = 1000
    lr = 0.0001
    memory_size = 50000
    qnet_update_frequency = 4
    gamma_q = 0.99
    gamma_e = 0.9 # 0.99 # 2 experiments to run here
