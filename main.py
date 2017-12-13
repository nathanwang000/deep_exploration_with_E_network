from lib.setting import *
from lib.model import selectNet
from lib.dataset import CartPoleVision, Pacwoman, MountainCar
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
    parser.add_argument('-g', '--game',
                        help='which game to play [pacwoman, mountain_car, cart_pole]',
                        choices=['pacwoman', 'mountain_car', 'cart_pole'],
                        default='cart_pole')
    
    args = parser.parse_args()
    return args

def op_run(env, run_name='default', plot=False,
           selection='softmax', setting=DefaultSetting()):
    selection = {'epsilon': epsilon_greedy(), 'softmax': softmax()}[selection]
    Qnet = selectNet('enet', env.name)

    t = Trainer(Qnet, env, selection, run_name=run_name, plot=plot,
                setting=setting)
    t.run()

def dqn_run(env, run_name='default', plot=False,
            selection='softmax', setting=DefaultSetting()):
    selection = {'epsilon': epsilon_greedy(), 'softmax': softmax()}[selection]
    Qnet = selectNet('dqn', env.name)

    t = Trainer(Qnet, env, selection, run_name=run_name, plot=plot,
                setting=setting)
    t.run()

def dora_run(env, run_name='default', plot=False,
             selection='softmax', setting=DefaultSetting()):
    selection = {'epsilon': LLL_epsilon_greedy(), 'softmax': LLL_softmax()}[selection]
    Qnet = selectNet('dqn', env.name)
    Enet = selectNet('enet', env.name)

    t = DoraTrainer(Qnet, Enet, env, selection, run_name=run_name,
                    plot=plot, setting=setting)
    t.run()

def run(args):
    env = {'cart_pole': CartPoleVision, 'pacwoman': Pacwoman,
           'mountain_car': MountainCar}[args.game]()
    setting = getSetting(args.game)

    if args.mode == 'dora':
        dora_run(run_name=args.name, plot=args.plot,
                 selection=args.selection, env=env,
                 setting=setting)
    elif args.mode == 'dqn':
        dqn_run(run_name=args.name, plot=args.plot,
                selection=args.selection, env=env,
                setting=setting)
    else: # optimistic start
        op_run(run_name=args.name, plot=args.plot,
               selection=args.selection, env=env,
               setting=setting) 
    
if __name__ == '__main__':
    args = parse_args()
    run(args)
