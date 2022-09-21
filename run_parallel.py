from utility.parallel_run import map_parallel
from main import run, parse_main_args
import copy


def parse_parrallel_args():
    parser = parse_main_args()
    parser.add_argument("-r", "--nruns", help="number of repeat", type=int, default=10)
    parser.add_argument(
        "-s",
        "--serial",
        help="serialy run the code",
        action="store_true",
        default=False,
    )
    return parser


if __name__ == "__main__":

    parser = parse_parrallel_args()
    args = parser.parse_args()
    nruns = args.nruns
    tasks = []

    for name in map(str, range(nruns)):
        args = copy.deepcopy(args)
        args.name = name
        tasks.append(args)

    if args.serial:
        for args in tasks:
            run(args)
    else:
        map_parallel(run, tasks)
