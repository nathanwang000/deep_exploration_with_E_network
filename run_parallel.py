from utility.parallel_run import map_parallel
from main import run, parse_args
import copy


class Args:
    def __init__(self, mode='dora', name='default', plot=False,
                 game='cart_pole', selection='softmax'):
        self.mode = mode
        self.name = name
        self.plot = plot
        self.game = game
        self.selection =selection
        
if __name__ == "__main__":

    nruns = 10
    args = parse_args()
    tasks = []

    for name in map(str, range(nruns)):
        args = copy.deepcopy(args)
        args.name = name
        tasks.append(args)
    
    map_parallel(run, tasks)
