from lib.setting import *
from lib.model import selectNet
from lib.dataset import CartPoleVision
from lib.action_selection import epsilon_greedy, LLL_epsilon_greedy
from lib.training import Trainer, DoraTrainer
import argparse

#parser = argparse.ArgumentParser()

def dqn_run():
    selection = epsilon_greedy()
    env = CartPoleVision()
    Qnet = selectNet()

    t = Trainer(Qnet, env, selection)
    t.run()

def dora_run():
    selection = LLL_epsilon_greedy()
    env = CartPoleVision()
    Qnet = selectNet()
    Enet = selectNet(Enet=True)
    
    t = DoraTrainer(Qnet, Enet, env, selection, lr=0.01)
    t.run()


dora_run()
# dqn_run()
