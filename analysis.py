import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import os
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="DORA training")
    parser.add_argument('-m', '--mode', choices=['multiple', 'combined', 'default'],
                        help='plot lines', default='default')
    args = parser.parse_args()
    return args

def plot_combined(logdir='logs'):
    colors = 'rbyok'
    count_helper = 0
    
    def helper(mode, title=None):
        nonlocal count_helper
        count_helper += 1
        
        fns = filter(lambda s: s.startswith(mode), os.listdir(logdir))
        data = []
        for fn in fns:
            log = joblib.load(os.path.join(logdir, fn))
            data.append(log)

        data = np.vstack(data)
        label = mode if not title else title
        sns.tsplot(data, condition=label, color=colors[count_helper])  

    sns.set(font_scale=1.5)
    helper('dora', "dora with LLL epsilon greedy")
    helper('dqn', "dqn with epsilon greedy")
    
    plt.xlabel("episode", fontsize=20)
    plt.ylabel("duration", fontsize=20)
    plt.legend()    
    plt.show()

def plot_multiple(mode='dora', logdir='logs'):
    fns = filter(lambda s: s.startswith(mode), os.listdir(logdir))
    for fn in fns:
        log = joblib.load(os.path.join(logdir, fn))
        plt.plot(log, label=fn.split('.')[0][len(mode)+1:])

    plt.title(mode, fontsize=20)
    plt.xlabel("episode", fontsize=20)
    plt.ylabel("duration", fontsize=20)
    plt.legend()
    plt.show()

def plot_default():
    dqn = joblib.load('logs/dqn_default.pkl')
    dora = joblib.load('logs/dora_default.pkl')

    plt.plot(dqn, label="dqn with epsilon greedy")
    plt.plot(dora, label="dora with LLL epsilon greedy")
    plt.legend()
    plt.xlabel("episode", fontsize=20)
    plt.ylabel("duration", fontsize=20)
    plt.show()

def run(args):
    if args.mode == 'combined':
        plot_combined()
    elif args.mode == 'default':
        plot_default()
    elif args.mode == 'multiple':
        plot_multiple('dora')
        plot_multiple('dqn')        

if __name__ == '__main__':
    args = parse_args()
    run(args)
