import numpy as np
import sys
from six import StringIO, b

from gym import utils
from gym.envs.toy_text import discrete

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "5": [
        "HHHHHHH",
        "LSBBBBG",
        "HHHHHHH"
    ],
    "15": [
        "HHHHHHHHHHHHHHHHH",
        "LSBBBBBBBBBBBBBBG",
        "HHHHHHHHHHHHHHHHH"
    ],
}

class BridgeEnv(discrete.DiscreteEnv):
    """
    Brdge environment is a grid world environment. You are on the left side of
    the river, and your aim is to go along the bridge to the otherside to get
    a reward of +10. If you go up or down on the bridge, you will fall into the
    river and get a reward of -100. If you go backwards, you will end your trip
    with reward of only +1.
    This environment is build upon the FrozenLake environment.
    A bridge environment with length=5 looks like this
        HHHHHHH
        LSBBBBG
        HHHHHHH
    S : starting point, safe
    L : terminating point, with low reward
    B : bridge, safe
    H : river
    G : goal, with high reward
    The episode ends when you reach a terminating state.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="5", noise=0):
        if desc is None and map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc,dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.noise = noise

        nA = 4
        nS = nrow * ncol #- 4 # Corners cannot be reached.

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row*ncol + col
        def inc(row, col, a):
            if a==0: # left
                col = max(col-1,0)
            elif a==1: # down
                row = min(row+1,nrow-1)
            elif a==2: # right
                col = min(col+1,ncol-1)
            elif a==3: # up
                row = max(row-1,0)
            return (row, col)

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b'LGH':
                        li.append((1.0, s, 0, True))
                    else:
                        for b in [(a-1)%4, a, (a+1)%4]:
                            newrow, newcol = inc(row, col, b)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            done = bytes(newletter) in b'LGH'
                            rew = 0.0
                            if newletter == b'L':
                    	        rew = 1.0
                            elif newletter == b'G':
                                rew = 10.0
                            elif newletter == b'H':
                                rew = -100.0
                            p = noise / 2
                            if b == a:
                            	p = 1 - noise
                            li.append((p, newstate, rew, done))

        super(BridgeEnv, self).__init__(nS, nA, P, isd)

    def _render(self, mode='human', close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left","Down","Right","Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            return outfile

class BridgeLargeEnv(BridgeEnv):

    def __init__(self):
        super(BridgeLargeEnv, self).__init__(map_name="15")

