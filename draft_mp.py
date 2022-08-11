#!/usr/bin/env python3
from functools import partial
from itertools import repeat
from multiprocessing import Pool, freeze_support
import numpy as np
import time



def do_stuff_1(x):
    return x * x

def do_stuff_2(mat, v):
    mat[v] = v*2

def main():
    func = do_stuff_2
    # a_args = [1,2,3]
    # second_arg = 1
    # with Pool() as pool:
    #     L = pool.starmap(func, [(1, 1), (2, 1), (3, 1)])
    #     M = pool.starmap(func, zip(a_args, repeat(second_arg)))
    #     N = pool.map(partial(func, b=second_arg), a_args)
    #     assert L == M == N
    mat = np.zeros(10)
    v_args = range(mat.shape[0])
    with Pool(4) as pool:
        pool.map(partial(func, mat=mat), v_args)


if __name__ == "__main__":
    freeze_support()
    main()





# from functools import partial
# from itertools import repeat
# from multiprocessing import Pool, freeze_support

# def func(a, b):
#     return a + b

# def main():
#     a_args = [1,2,3]
#     second_arg = 1
#     with Pool() as pool:
#         L = pool.starmap(func, [(1, 1), (2, 1), (3, 1)])
#         M = pool.starmap(func, zip(a_args, repeat(second_arg)))
#         N = pool.map(partial(func, b=second_arg), a_args)
#         assert L == M == N

# if __name__=="__main__":
#     freeze_support()
#     main()