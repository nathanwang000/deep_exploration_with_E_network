from setting import *
import numpy as np
import math

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
    
