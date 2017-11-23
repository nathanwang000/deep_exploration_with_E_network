import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import os

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

    
plot_multiple('dora')
plot_multiple('dqn')
