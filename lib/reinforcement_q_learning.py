from setting import *
from model import DQN
from dataset import CartPoleVision
from action_selection import epsilon_greedy
from training import Trainer

selection = epsilon_greedy()
env = CartPoleVision()
model = DQN()

t = Trainer(model, env, selection)
t.run()

