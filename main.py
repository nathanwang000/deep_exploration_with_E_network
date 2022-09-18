import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # "0, 1" for multiple

from lib.setting import *
from lib.model import selectNet
from lib.dataset import CartPoleVision, Pacwoman, MountainCar,MountainCarLong, Bridge
from lib.action_selection import epsilon_greedy, LLL_epsilon_greedy, softmax, LLL_softmax
from lib.training import Trainer, DoraTrainer
import joblib
import argparse


def parse_main_args():
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
                        help='which game to play [pacwoman, mountain_car, mountain_car_long cart_pole]',
                        choices=['pacwoman', 'mountain_car','mountain_car_long', 'cart_pole', 'bridge'],
                        default='cart_pole')
    parser.add_argument('-l', '--logpath',
                        help='where to save the log, defualt to logs',
                        default='logs')

    parser.add_argument('-t', '--temperature',
                        help='Temperature for softmax',
                        default=1.0,type=float)


    return parser

def op_run(env, run_name='default', plot=False,
           selection='softmax', setting=DefaultSetting(),
           log_path='logs'):
    selection = {'epsilon': epsilon_greedy(), 'softmax': softmax()}[selection]
    Qnet = selectNet('enet', env.name)

    t = Trainer(Qnet, env, selection, run_name=run_name, plot=plot,
                setting=setting, log_path=log_path)
    t.run()

def dqn_run(env, run_name='default', plot=False,
            selection='softmax', setting=DefaultSetting(),
            log_path='logs',temperature=1.0):
    selection = {'epsilon': epsilon_greedy(), 'softmax': softmax(T=temperature)}[selection]
    Qnet = selectNet('dqn', env.name)

    t = Trainer(Qnet, env, selection, run_name=run_name, plot=plot,
                setting=setting, log_path=log_path)
    t.run()

def dora_run(env, run_name='default', plot=False,
             selection='softmax', setting=DefaultSetting(),
             log_path='logs',temperature=1.0):
    selection = {'epsilon': LLL_epsilon_greedy(), 'softmax': LLL_softmax(T=temperature)}[selection]
    Qnet = selectNet('dqn', env.name)
    Enet = selectNet('enet', env.name)

    t = DoraTrainer(Qnet, Enet, env, selection, run_name=run_name,
                    plot=plot, setting=setting, log_path=log_path)
    t.run()

def run(args):
    env = {'cart_pole': CartPoleVision, 'pacwoman': Pacwoman,
           'mountain_car': MountainCar, 'bridge': Bridge,'mountain_car_long':MountainCarLong}[args.game]()
    setting = getSetting(args.game)
    log_path = set_log_path(args.logpath)

    if args.mode == 'dora':
        dora_run(run_name=args.name, plot=args.plot,
                 selection=args.selection, env=env,
                 setting=setting, log_path=log_path, temperature=args.temperature)
    elif args.mode == 'dqn':
        dqn_run(run_name=args.name, plot=args.plot,
                selection=args.selection, env=env,
                setting=setting, log_path=log_path, temperature=args.temperature)
    else: # optimistic start
        op_run(run_name=args.name, plot=args.plot,
               selection=args.selection, env=env,
               setting=setting, log_path=log_path) 
    
if __name__ == '__main__':
    parser = parse_main_args()
    args = parser.parse_args()
    run(args)

    if args.game == 'bridge' and args.mode == 'dora':
        joblib.dump(EsCounter,
                    os.path.join(args.logpath,
                                 "counter_dora_%s.pkl" % args.name))
    
