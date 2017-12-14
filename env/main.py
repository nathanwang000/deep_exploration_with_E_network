import numpy as np
import gym
import matplotlib.pyplot as plt
import matplotlib
from algorithms import *

def generate_plot_1():
    env = gym.make("BridgeEnv-v0").unwrapped
    marker_style=[':','-.','--','-',':','-.','--','-',':','-.']
    num_episodes=10000
    seed = [175,75,281,170,264,298,284, 144,38,20,116,168,277,49,248,56,118,274,161,136,86,187,55,279,215,
            250,149,289,253,35,7,207,93,259,285,205,169,45,296,245,274,206,91,220,83,185,89,156,222,224]
    MSE = np.zeros((num_episodes))
    for s in seed:
        MSE = MSE + lll_learning(env, alpha=0.35, epsilon=0.6, num_episodes=num_episodes, seed = s)[0]
    MSE = MSE / len(seed)
    font = {'size' : 13}
    plt.figure()
    matplotlib.rc('font', **font)
    plt.plot(MSE, label="No bonus")
    for i in range(10):
        MSE = np.zeros((num_episodes))
        for s in seed:
            MSE = MSE + lll_counter(env, alpha=0.35, epsilon=0.6, num_episodes=num_episodes,
                             gamma_e=0.1*i, seed=s)
        MSE = MSE/len(seed)
        plt.plot(MSE, marker_style[i],label=r"$\gamma_e$ =%.1f" % (0.1*i))
    plt.legend()
    plt.axis([1,num_episodes,0,360])
    plt.xlabel("Episodes")
    plt.ylabel("MSE")
    plt.semilogx()
    #plt.show()

def generate_plot_2():
    seed = [175,75,281,170,264,298,284, 144,38,20,116,168,277,49,248,56,118,274,161,136,86,187,55,279,215,
            250,149,289,253,35,7,207,93,259,285,205,169,45,296,245,274,206,91,220,83,185,89,156,222,224]
    env = gym.make("BridgeLargeEnv-v0").unwrapped
    num_episodes=10000
    MSE=[]
    for i in range(8):
        MSE.append(np.zeros((num_episodes)))
    for s in seed:
        MSE[0] = MSE[0] + lll_learning(env, alpha=0.2, epsilon=0.9, num_episodes=num_episodes, seed=s)[0]
        MSE[2] = MSE[2] + lll_learning(env, alpha=0.5, epsilon=0.9, num_episodes=num_episodes, isDora=True,
                      gamma_e=0.9, seed=s)[0]
        MSE[6] = MSE[6] + lll_counter(env, alpha=0.5, epsilon=0.9, num_episodes=num_episodes, seed=s)
        MSE[3] = MSE[3] + lll_learning(env, alpha=0.5, epsilon=0.9, num_episodes=num_episodes, isDora=True,
                          a_selection="softmax", gamma_e=0.9, seed=s)[0]
        MSE[4] = MSE[4] + lll_learning(env, alpha=0.5, epsilon=0.9, num_episodes=num_episodes,
                      a_selection="softmax", seed=s)[0]
        MSE[5] = MSE[5] + lll_counter(env, alpha=0.5, epsilon=0.9, num_episodes=num_episodes,
                                    a_selection="softmax", seed=s)
        MSE[1] = MSE[1] + lll_learning(env, alpha=0.2, epsilon=0.9, num_episodes=num_episodes, a_selection="UCB", c=2, seed=s)[0]
        MSE[7] = MSE[7] + lll_counter(env, alpha=0.5, epsilon=0.9, num_episodes=num_episodes, 
                         a_selection="UCB", seed=s)
    for i in range(8):
        MSE[i] = MSE[i] / len(seed)
    font = {'size' : 13}
    plt.figure()
    matplotlib.rc('font', **font)
    plt.plot(MSE[0], ':',label="e-greedy")
    plt.plot(MSE[2], '-.',label="LLL e-greedy(E-values)")
    plt.plot(MSE[6], '--',label="LLL egreedy(Counters)")
    plt.plot(MSE[3], '-',label="LLL softmax(E-values)")
    plt.plot(MSE[4], ':',label="softmax")
    plt.plot(MSE[5], '--',label="LLL softmax(Counters)")
    plt.plot(MSE[1], '-.',label="UCB")
    plt.plot(MSE[7], '-',label="UCB(Evalues)")
    plt.legend()
    plt.axis([1,3000,0,510])
    plt.xlabel("Episodes")
    plt.ylabel("MSE")
    #plt.show()

def generate_plot_3_4():
    marker_style=["o","s","^","v","P"]
    env = gym.make("BridgeEnv-v0").unwrapped    
    num_episodes=50
    seed = 42
    (MSE, res_E, res_N, res_Q) = lll_learning(env, alpha=0.5, epsilon=0.9, num_episodes=num_episodes, isDora=True,
                          a_selection="softmax", gamma_e=0.9, seed=seed)
    s = (8,9,10,11,12)
    a = 2
    plt.figure()
    font = {'size' : 13}
    matplotlib.rc('font', **font)
    for k in range(5):
        C = np.zeros(num_episodes)
        info = np.zeros(num_episodes)
        for i in range(num_episodes):
            C[i] = res_N[i][s[k]][a]
            info[i] = res_Q[i][s[k]-8][a]
        plt.plot(C, info, marker_style[k], label="C(%d,%d)" % (s[k], a))
    C = np.zeros(num_episodes)
    for i in range(num_episodes):
        C[i] = res_N[i][9][0]
        info[i] = res_Q[i][9-8][0]
    plt.plot(C, info, "*", label="C(%d,%d)" % (9, 0))
    plt.legend()
    plt.xlabel("C(s,a)")
    plt.ylabel(r"$|\frac{Q-Q*}{Q*}|$")
    plt.figure()
    font = {'size' : 13}
    matplotlib.rc('font', **font)
    for k in range(5):
        C = np.zeros(num_episodes)
        info = np.zeros(num_episodes)
        for i in range(num_episodes):
            C[i] = res_E[i][s[k]][a]
            info[i] = res_Q[i][s[k]-8][a]
        plt.plot(C, info, marker_style[k], label=r"$\log_{1-\alpha}(E(%d,%d))$" % (s[k], a))
    C = np.zeros(num_episodes)
    for i in range(num_episodes):
        C[i] = res_E[i][9][0]
        info[i] = res_Q[i][9-8][0]
    plt.plot(C, info, "*", label=r"$\log_{1-\alpha}(E(%d,%d))$" % (9, 0))
    legend=plt.legend()
    legend.get_title().set_fontsize(fontsize = 20)
    plt.xlabel(r"$\log_{1-\alpha}(E(s,a))$")
    plt.ylabel(r"$|\frac{Q-Q*}{Q*}|$")
    #plt.show()

def generate_plot_5():
    env = gym.make("BridgeLargeEnv-v0").unwrapped
    num_episodes = 1000
    s = 25
    MSE = lll_learning(env, alpha=0.5, epsilon=0.9, num_episodes=num_episodes, isDora=True,a_selection="softmax", gamma_e=0.9, seed=s)[0]
    MSE = MSE / np.amax(MSE)
    MSE2 = delayed_Q_learning(env, gamma=0.9, m=10, epsilon1=0.01, num_episodes=num_episodes, seed=s)
    MSE2 = MSE2 / np.amax(MSE2)
    plt.figure()
    font = {'size' : 13}
    matplotlib.rc('font', **font)
    plt.plot(MSE, '-',label="LLL softmax(E-values)")
    plt.plot(MSE2, '-.',label="Delayed Q learning")
    plt.legend()
    plt.axis([1,600,0,1])
    plt.xlabel("Episodes")
    plt.ylabel("MSE")
    #plt.show()

if __name__ == "__main__":
    #generate_plot_1()
    #generate_plot_2()
    #generate_plot_3_4()
    generate_plot_5()
    plt.show()