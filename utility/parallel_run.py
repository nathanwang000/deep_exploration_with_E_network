import multiprocessing
import numpy as np
import time


def iterable(a):
    try:
        iter(a)
        return True
    except:
        return False


def map_parallel(f, tasks):
    """
    embrassingly parallel
    f: function to apply
    tasks: list of argument lists
    """
    n_cpus = min(int(multiprocessing.cpu_count() / 2), len(tasks))

    result_list = []
    pool = multiprocessing.Pool(n_cpus)
    for task in tasks:
        if not iterable(task):
            task = (task,)
        result_list.append(pool.apply_async(func=f, args=task))
    while True:
        try:

            def call_if_ready(result):
                if result.ready():
                    result.get()
                    return True
                else:
                    return False

            done_list = list(map(call_if_ready, result_list))
            print("{}/{} done".format(sum(done_list), len(result_list)))
            if np.all(done_list):
                break
            time.sleep(3)
        except:
            pool.terminate()
            raise
    print("finished preprocessing")

    return [r.get() for r in result_list]
