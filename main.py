from lib.setting import *
from lib.model import selectNet
from lib.dataset import CartPoleVision
from lib.action_selection import epsilon_greedy, LLL_epsilon_greedy, softmax, LLL_softmax
from lib.training import Trainer, DoraTrainer
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="DORA training")
    parser.add_argument('-m', '--mode', choices=['dqn', 'dora', 'op'],
                        help='dqn or dora', default='dora')
    parser.add_argument('-n', '--name', help='name to save', default='default')
    parser.add_argument('-p', '--plot', action="store_true", help='plot or not',
                        default=False)
    parser.add_argument('-a', '--selection', help='action selection rule',
                        choices=['epsilon', 'softmax'],
                        default='softmax')
    
    args = parser.parse_args()
    return args

def op_run(run_name='default', plot=False, selection='softmax'):
    selection = {'epsilon': epsilon_greedy(), 'softmax': softmax()}[selection]
    env = CartPoleVision()
    Qnet = selectNet('enet')

    t = Trainer(Qnet, env, selection, run_name=run_name, plot=plot)
    t.run()

def dqn_run(run_name='default', plot=False, selection='softmax'):
    selection = {'epsilon': epsilon_greedy(), 'softmax': softmax()}[selection]
    env = CartPoleVision()
    Qnet = selectNet('dqn')

    t = Trainer(Qnet, env, selection, run_name=run_name, plot=plot)
    t.run()

def dora_run(run_name='default', plot=False, selection='softmax'):
    selection = {'epsilon': LLL_epsilon_greedy(), 'softmax': LLL_softmax()}[selection]
    env = CartPoleVision()
    Qnet = selectNet()
    Enet = selectNet('enet')
    
    t = DoraTrainer(Qnet, Enet, env, selection, lr=0.01, run_name=run_name, plot=plot)
    t.run()

def run(args):
    if args.mode == 'dora':
        dora_run(run_name=args.name, plot=args.plot, selection=args.selection)
    elif args.mode == 'dqn':
        dqn_run(run_name=args.name, plot=args.plot, selection=args.selection)
    else: # optimistic start
        op_run(run_name=args.name, plot=args.plot, selection=args.selection)        

    
if __name__ == '__main__':
    args = parse_args()
    run(args)
