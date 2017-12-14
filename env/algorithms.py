import numpy as np
from gym.envs.my_env import bridge
import gym
import matplotlib.pyplot as plt
import matplotlib

def value_iteration(env, gamma = 0.9, theta = 0.0001):
    ''' Performs value iteration for the given environment.
        If there's a tie, use the first occurrence of max value.
    
    :param env: Unwrapped OpenAI gym environment.
    :param gamma: Decay rate parameter.
    :param theta: Acceptable convergence rate
    :return: Policy, a |State|x|Action| matrix of the probability
        of taking an action in a given state.
    '''
    #np.random.seed(42)
    P = env.P
    nS = env.nS
    nA = env.nA
    V= np.zeros(nS)
    policy = np.zeros((nS, nA))
    while 1:
        delta = 0
        lastV = np.zeros(nS)
        for s in range(nS):
            lastV[s] = V[s]
        for s in range(nS):
            maxValue = -1000
            maxA = -1
            for a in range(nA):
                tempSum = 0
                for tup in P[s][a]:
                    tempSum = tempSum + tup[0] * (tup[2] + gamma * lastV[tup[1]])
                if tempSum > maxValue:
                    maxValue = tempSum
                    maxA = a
            V[s] = maxValue
            policy[s] = np.zeros(nA)
            if maxA != -1:
                policy[s][maxA] = 1
            delta = np.max([delta, np.abs(lastV[s] - V[s])])
        if delta < theta:
            break
    print(V)
    return policy

def lll_learning(env, alpha = 0.2, gamma = 0.9,epsilon = 0.1,
               num_episodes = 500, isDora=False,a_selection="egreedy",
               gamma_e=0, c=2, seed=None):
    ''' Performs Q-learning or Dora algorithm for the given environment.

    :param env: Unwrapped OpenAI gym environment.
    :param alpha: Learning rate parameter.
    :param gamma: Decay rate parameter.
    :param num_episodes: Number of episodes to use for learning.
    :param isDora: Dora or Q-learning.
    :param a_selection: Action-selection rule. Choose one from egreedy, UCB, and softmax.
    :param gamma_e: Decay rate for E-values.
    :param c: Parameter for UCB.
    :param seed: Random seed.
    :return: MSE.
    '''
    print(isDora, gamma_e, seed)
    nS = env.nS
    nA = env.nA
    res_N = []
    res_E = []
    res_Q = []
    Q_true = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
                   [0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
                   [1.,-100.,6.561,-100.],[0.9,-100.,7.29,-100.],
                   [0.81,-100.,8.1,-100.],[0.729,-100.,9.,-100.],
                   [0.6561,-100.,10.,-100.],[0,0,0,0],[0,0,0,0],[0,0,0,0],
                   [0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
                   [0,0,0,0]])
    p_optimal = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
                          [0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
                          [0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],
                          [0,0,1,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
                          [0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
                          [0,0,0,0]])
    if env.nS == 51:
        Q_true = np.zeros((nS, nA))
        for i in range(15):
            Q_true[i+18][0] = 1.0*(0.9**i)
            Q_true[i+18][1] = -100.
            Q_true[i+18][2] = 10.0*(0.9**(14-i))
            Q_true[i+18][3] = -100.
        p_optimal = np.zeros((nS, nA))
        for i in range(15):
            p_optimal[i+18][2] = 1
    MSE = np.zeros(num_episodes)
    if not seed == None:
        np.random.seed(seed)
    t = 0
    Q = np.zeros((nS, nA))
    N = np.zeros((nS, nA))
    if isDora:
        E = np.ones((nS, nA))
        E = E * 0.999999
    policy = get_policy(nA, nS, Q, epsilon)
    for n_episode in range(num_episodes):
        s = env.reset()
        if isDora:
            as_rule = policy[s]
            if a_selection == "UCB":
                as_rule = ucb(Q[s], c, N[s], t)
            elif a_selection == "softmax":
                as_rule = softmax(Q[s], T=0.1)
            if 0 in as_rule:
                rule = as_rule
            else:
                rule = np.log(as_rule)-np.log(np.log(E[s]) / np.log(1-alpha))
            a = np.argmax(rule)
        for step in range(100):
            t += 1
            policy = get_policy(nA, nS, Q, epsilon)
            if not isDora:
                if a_selection=="egreedy":
                    a = np.random.choice(nA, 1, p = policy[s])[0]
                elif a_selection=="UCB":
                    a = np.argmax(ucb(Q[s], c, N[s], t))
                elif a_selection=="softmax":
                    a = np.random.choice(nA, 1, p = softmax(Q[s], T=1))[0]
            next_s, r, done, _ = env.step(a)
            N[s][a] += 1
            #print(s, a)
            if isDora:
                Q[s][a] = (1-alpha)*Q[s][a] + alpha * (r + gamma * np.amax(Q[next_s]))
                as_rule = policy[next_s]
                if a_selection == "UCB":
                    as_rule = ucb(Q[next_s], c, N[next_s], t)
                elif a_selection == "softmax":
                    as_rule = softmax(Q[next_s], T=1);
                if 0 in as_rule:
                    rule = as_rule
                else:
                    rule = np.log(as_rule)-np.log(np.log(E[next_s]) / np.log(1-alpha))
                
                next_a = np.argmax(rule)
                E[s][a]=(1-alpha) * E[s][a] + alpha * gamma_e * E[next_s][next_a]
                a = next_a
            else:
                Q[s][a] = Q[s][a] + alpha * (r + gamma * np.amax(Q[next_s]) - Q[s][a])
            if done:
                if isDora:
                    E[next_s][next_a] = (1-alpha) * E[next_s][next_a]
                break
            s = next_s
            
        MSE[n_episode] = np.sum(p_optimal * np.square(Q - Q_true))
        if isDora:
            res_E.append(np.log(E) / np.log(1-alpha))
            res_N.append(N + np.zeros((nS, nA)))
            res_Q.append(np.abs((Q[8:13] - Q_true[8:13]) / Q_true[8:13]))
    return MSE, res_E, res_N, res_Q


def lll_counter(env, alpha = 0.5, gamma = 0.9,
               epsilon = 0.1, num_episodes = 500, a_selection="egreedy", gamma_e=0, c=2, seed=None):
    ''' Performs Q-learning with generalized counter for the given environment.
    :param env: Unwrapped OpenAI gym environment.
    :param alpha:Learning rate parameter.
    :param gamma: Decay rate parameter.
    :param num_episodes: Number of episodes to use for learning.
    :param epsilon: Probability of taking a random move.
    :param a_selection: Action-selection rule. Choose one from egreedy, UCB, and softmax.
    :param gamma_e: Decay rate for E-values.
    :param c: Parameter for UCB.
    :param seed: Random seed.
    :return: Mean Squared Error of Q-value along the number of episodes.
    '''
    print(a_selection, gamma_e, seed)
    nS = env.nS
    nA = env.nA
    Q_true = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
                   [0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
                   [1.,-100.,6.561,-100.],[0.9,-100.,7.29,-100.],
                   [0.81,-100.,8.1,-100.],[0.729,-100.,9.,-100.],
                   [0.6561,-100.,10.,-100.],[0,0,0,0],[0,0,0,0],[0,0,0,0],
                   [0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
                   [0,0,0,0]])
    p_optimal = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
                          [0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
                          [0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],
                          [0,0,1,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
                          [0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
                          [0,0,0,0]])
    if env.nS == 51:
        Q_true = np.zeros((nS, nA))
        for i in range(15):
            Q_true[i+18][0] = 1.0*(0.9**i)
            Q_true[i+18][1] = -100.
            Q_true[i+18][2] = 10.0*(0.9**(14-i))
            Q_true[i+18][3] = -100.
        p_optimal = np.zeros((nS, nA))
        for i in range(15):
            p_optimal[i+18][2] = 1
    np.random.seed(42)
    MSE = np.zeros(num_episodes)
    Q = np.zeros((nS, nA))
    N = np.zeros((nS, nA))
    E = np.ones((nS, nA)) * 0.999999
    t = 0
    if not seed == None:
        np.random.seed(seed)
    for n_episode in range(num_episodes):
        s = env.reset()
        if a_selection == "egreedy":
            policy = get_policy(nA, nS, Q, epsilon, isEvalue=True, E=E, alpha=alpha)
            a = np.random.choice(nA, 1, p = policy[s])[0]
        elif a_selection == "UCB":
            a = np.argmax(ucb(Q[s], c, N[s], t))
        elif a_selection == "softmax":
            a = np.random.choice(nA, 1, p = softmax(Q[s], T=1))[0]
        for step in range(100):
            t += 1
            next_s, r, done, _ = env.step(a)
            N[s][a] += 1
            Q[s][a] = Q[s][a] + alpha * (r + gamma * np.amax(Q[next_s]) - Q[s][a])
            
            if a_selection == "egreedy":
                policy = get_policy(nA, nS, Q, epsilon, isEvalue=True, E=E, alpha=alpha)
                next_a = np.random.choice(nA, 1, p = policy[s])[0]
            elif a_selection == "UCB":
                as_rule = ucb(Q[next_s], c, N[next_s], t, isEvalue=True, E=E[s], alpha=alpha)
                next_a = np.argmax(as_rule)
            elif a_selection == "softmax":
                next_a = np.random.choice(nA, 1, p = softmax(Q[next_s], T=1))[0]
            E[s][a]=(1-alpha) * E[s][a] + alpha * gamma_e * E[next_s][next_a]
            a = next_a
            
            if done:
                E[next_s][next_a] = (1-alpha) * E[next_s][next_a]
                break
            s = next_s
        MSE[n_episode] = np.sum(p_optimal * np.square(Q - Q_true))
    return MSE

def delayed_Q_learning(env, gamma, m, epsilon1, num_episodes=30, seed=None):
    ''' Performs delayed Q-learning.
    :param env: Unwrapped OpenAI gym environment.
    :param gamma: Decay rate parameter.
    :param num_episodes: Number of episodes to use for learning.
    :param epsilon1: Parameter for mixing time.
    :param seed: Random seed.
    :return: Mean Squared Error of Q-value along the number of episodes.
    '''
    nS = env.nS
    nA = env.nA
    Q_true = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
                   [0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
                   [1.,-100.,6.561,-100.],[0.9,-100.,7.29,-100.],
                   [0.81,-100.,8.1,-100.],[0.729,-100.,9.,-100.],
                   [0.6561,-100.,10.,-100.],[0,0,0,0],[0,0,0,0],[0,0,0,0],
                   [0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
                   [0,0,0,0]])
    p_optimal = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
                          [0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
                          [0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],
                          [0,0,1,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
                          [0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
                          [0,0,0,0]])
    if env.nS == 51:
        Q_true = np.zeros((nS, nA))
        for i in range(15):
            Q_true[i+18][0] = 1.0*(0.9**i)
            Q_true[i+18][1] = -100.
            Q_true[i+18][2] = 10.0*(0.9**(14-i))
            Q_true[i+18][3] = -100.
        p_optimal = np.zeros((nS, nA))
        for i in range(15):
            p_optimal[i+18][2] = 1
    MSE = np.zeros(num_episodes)
    if not seed == None:
        np.random.seed(seed)
    Q = np.ones((nS, nA))*(1/(1-gamma))
    U = np.zeros((nS, nA))
    l = np.zeros((nS, nA))
    tsa = np.zeros((nS, nA))
    LEARN = []
    for i in range(nS):
        LEARN.append([True]*nA)
    t_star = 0
    t = 0
    for n_episode in range(num_episodes):
        s = env.reset()
        for step in range(100):
            t += 1
            a = np.argmax(Q[s])
            next_s, r, done, _ = env.step(a)
            if LEARN[s][a] == True:
                U[s][a] += r + gamma*np.amax(Q[next_s])
                l[s][a] += 1
                if l[s][a] == m:
                    if (Q[s][a]-U[s][a]/m) >= 2*epsilon1:
                        Q[s][a] = U[s][a]/m + epsilon1
                        t_star = t
                    elif tsa[s][a] >= t_star:
                        LEARN[s][a] = False
                    tsa[s][a] = t
                    U[s][a] = 0
                    l[s][a] = 0
            elif tsa[s][a] < t_star:
                LEARN[s][a] = True
            s = next_s
        MSE[n_episode] = np.sum(p_optimal * np.square(Q - Q_true))
    return MSE
    
    

def get_policy(nA, nS, Q, epsilon, isEvalue=False, E=None, alpha=0):
    ''' Derive an epsilon-soft policy from Q. '''
    policy = np.ones((nS, nA)) * epsilon / nA
    rule = np.zeros((nS, nA))
    for s in range(nS):
        if isEvalue:
            exp_bonus = 1 / (np.log(E[s]) / np.log(1-alpha))
            rule[s] = Q[s] + exp_bonus
        else:
            rule[s] = Q[s]
        maxA = np.argmax(rule[s])
        policy[s][maxA] = policy[s][maxA] + 1.0 - epsilon
    return policy

def ucb(Q, c, N, t, isEvalue=False, E=None, alpha=0):
    ''' Q is Q[s]. N is N[s]. E is E[s]. s is the current state. '''
    rule = np.zeros(4)
    for i in range(4):
        if N[i] == 0:
            rule = np.zeros(4)
            rule[i] = 1
            break
        else:
            rule[i] = Q[i] + c * np.square(np.log(t) / N[i])
            if isEvalue:
                exp_bonus = 1 / (np.log(E[i]) / np.log(1-alpha))
                rule[i] = rule[i] + exp_bonus
    return rule

def softmax(Q, T, isEvalue=False, E=None, alpha=0):
    ''' Q is Q[s]. E is E[s]. '''
    if isEvalue:
        exp_bonus = 1 / (np.log(E) / np.log(1-alpha))
        newQ = Q+exp_bonus
    else:
        newQ = Q
    return np.exp(T*newQ) / np.sum(np.exp(T*newQ))

