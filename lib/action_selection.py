from lib.setting import *
import numpy as np
import math

def safe_log(values, base=math.e):
    eps = 1e-10 # avoid zero in log
    if sum(values < 1e-8) > 0:
        values += eps

    return np.log(values) / np.log(base)
    
class epsilon_greedy:
    def __init__(self):
        self.steps_done = 0
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 200

    def select_action(self, Qs):
        eps = self.eps_end + (self.eps_start - self.eps_end) * \
              math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        if np.random.rand() < eps:
            action = int(np.random.choice(len(Qs)))
        else:
            action = int(np.argmax(Qs))
        return LongTensor([[action]])

class LLL_epsilon_greedy:
    def __init__(self):
        self.steps_done = 0
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 200

    def q2pr(self, Qs):
        # qvalues to probabilities
        eps = self.eps_end + (self.eps_start - self.eps_end) * \
              math.exp(-1. * self.steps_done / self.eps_decay)
        num_a = len(Qs)
        prob = np.ones_like(Qs) * eps / num_a
        prob[np.argmax(Qs)] += 1-eps
        return prob
        
    def select_action(self, Qs, Es, lr):
        # log f(Qs) - log log_{1-lr} Es
        prob = self.q2pr(Qs)
        logF = safe_log(prob)
        loglogEs = safe_log(safe_log(Es, 1-lr))
        self.steps_done += 1

        action = int(np.argmax(logF - loglogEs))
        # print("logEs", safe_log(Es, 1-lr))
        # print('Es', Es, '1-lr', 1-lr)
        # print(logF.max(), loglogEs.max())

        return LongTensor([[action]])
    
    
