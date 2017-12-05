from utility.parallel_run import map_parallel
from main import run
import copy

class Args:
    def __init__(self, mode='dora', name='default', plot=False):
        self.mode = mode
        self.name = name
        self.plot = plot
        
if __name__ == "__main__":

    tasks = []

    for mode in ['dora', 'dqn']:
        for name in map(str, range(10)):
            plot = False
            tasks.append(Args(mode, name, plot))
    
    map_parallel(run, tasks)
