import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import os
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="DORA training")
    parser.add_argument(
        "-m", "--mode", choices=["dqn", "dora"], help="dqn or dora?", default="dora"
    )
    parser.add_argument(
        "-p",
        "--ptype",
        choices=["multiple", "combined", "default"],
        help="plot type",
        default="default",
    )
    parser.add_argument(
        "-l", "--logpath", help="where to save the log, defualt to logs", default="logs"
    )
    args = parser.parse_args()
    return args


def plot_combined(mode="dora", logdir="logs"):
    colors = "rbyok"
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
    helper(mode, mode)

    plt.xlabel("episode", fontsize=20)
    plt.ylabel("rewards", fontsize=20)
    plt.legend()
    plt.show()


def plot_multiple(mode="dora", logdir="logs"):
    fns = filter(lambda s: s.startswith(mode), os.listdir(logdir))
    for fn in fns:
        log = joblib.load(os.path.join(logdir, fn))
        plt.plot(log, label=fn.split(".")[0][len(mode) + 1 :])

    plt.title(mode, fontsize=20)
    plt.xlabel("episode", fontsize=20)
    plt.ylabel("rewards", fontsize=20)
    plt.legend()
    plt.show()


def plot_default(mode, logdir="logs"):
    fn = joblib.load("{}/{}_default.pkl".format(logdir, mode))

    plt.plot(fn, label=mode)
    plt.legend()
    plt.xlabel("episode", fontsize=20)
    plt.ylabel("rewards", fontsize=20)
    plt.show()


def run(args):
    mode = args.mode
    logdir = args.logpath
    if args.ptype == "combined":
        plot_combined(mode, logdir)
    elif args.ptype == "default":
        plot_default(mode, logdir)
    elif args.ptype == "multiple":
        plot_multiple(mode, logdir)


if __name__ == "__main__":
    args = parse_args()
    run(args)
