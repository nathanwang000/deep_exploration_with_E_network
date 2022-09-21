import numpy as np
import gym
from collections import defaultdict


def value_iteration(env, gamma=0.95, theta=0.0001):
    """Performs value iteration for the given environment.

    Initialize the values to all zeros.

    :param env: Unwrapped OpenAI gym environment.
    :param gamma: Decay rate parameter.
    :param theta: Acceptable convergence rate.
    :return: V
    """

    V = np.zeros(env.nS)
    delta = theta + 1
    while delta >= theta:
        delta = 0
        V_old = V
        V = np.zeros(env.nS)
        for s in range(env.nS):
            isFirst = True
            for a in range(env.nA):
                inner = 0
                for p, s_prime, r, done in env.P[s][a]:
                    inner += p * (r + gamma * V_old[s_prime])
                if isFirst:
                    isFirst = False
                    V[s] = inner
                else:
                    V[s] = max(inner, V[s])
            delta = max(delta, abs(V_old[s] - V[s]))
            V_old[s] = V[s]
    return V


def q_value_iteration(env, gamma=0.95, theta=1e-10):
    """Performs value iteration for the given environment.

    Initialize the values to all zeros.

    :param env: Unwrapped OpenAI gym environment.
    :param gamma: Decay rate parameter.
    :param theta: Acceptable convergence rate.
    :return: Q table
    """

    Q = np.zeros([env.nS, env.nA])
    delta = theta + 1
    while delta >= theta:
        delta = 0
        Q_old = Q
        Q = np.zeros([env.nS, env.nA])
        for s in range(env.nS):
            isFirst = True
            for a in range(env.nA):
                for p, s_prime, r, done in env.P[s][a]:
                    Q[s, a] += p * (
                        r + gamma * max([Q_old[s_prime, a_] for a_ in range(env.nA)])
                    )
                delta = max(delta, abs(Q_old[s, a] - Q[s, a]))
            Q_old[s, a] = Q[s, a]
    return Q


def test():
    env = gym.make("BridgeEnv-v0").unwrapped
    print(q_value_iteration(env))
