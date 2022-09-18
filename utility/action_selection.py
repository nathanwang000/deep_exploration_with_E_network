import numpy as np

# finite action setting
def softmax(Qs):
    # Qs is Q(s,*), a |a| dimensional vector
    prob = np.exp(Qs) / np.exp(Qs).sum()
    return np.random.choice(len(Qs), p=prob)


def eps_greedy(Qs, eps=0.1):
    # choose randomly with pr eps, else choose max_a Qs(a)
    if np.random.rand() < eps:
        return np.random.choice(len(Qs))
    return np.argmax(Qs)


def UCB(Qs, na, c=2):
    # Qs is Q(s,*), a |a| dimensional vector
    # na is number of time visited action a, a |a| dimensional vector
    # c a hyperparm controling tradeoff between explore and exploit
    Qs, na = np.array(Qs), np.array(na)
    t = na.sum()
    return np.argmax(Qs + np.sqrt(t / na))
