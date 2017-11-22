from lib.setting import *
from lib.model import selectNet
from lib.dataset import CartPoleVision
from lib.action_selection import epsilon_greedy, LLL_epsilon_greedy
from lib.training import Trainer, DoraTrainer
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="DORA training")
    parser.add_argument('-m', '--mode', choices=['dqn', 'dora'],
                        help='dqn or dora', default='dora')
    parser.add_argument('-n', '--name', help='name to save', default='default')
    parser.add_argument('-p', '--plot', action="store_true", default=False)
    args = parser.parse_args()
    return args

def dqn_run(run_name='default', plot=False):
    selection = epsilon_greedy()
    env = CartPoleVision()
    Qnet = selectNet()

    t = Trainer(Qnet, env, selection, run_name=run_name, plot=plot)
    t.run()

def dora_run(run_name='default', plot=False):
    selection = LLL_epsilon_greedy()
    env = CartPoleVision()
    Qnet = selectNet()
    Enet = selectNet(Enet=True)
    
    t = DoraTrainer(Qnet, Enet, env, selection, lr=0.01, run_name=run_name, plot=plot)
    t.run()

if __name__ == '__main__':
    args = parse_args()

    if args.mode == 'dora':
        dora_run(run_name=args.name, plot=args.plot)
    else:
        dqn_run(run_name=args.name, plot=args.plot)
