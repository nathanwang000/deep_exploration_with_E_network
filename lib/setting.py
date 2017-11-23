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

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

