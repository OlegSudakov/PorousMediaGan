import pickle
from os.path import isfile
import numpy as np
from copy import copy
import multiprocessing as mp

path = "../../data/dicts/comb_to_update_new.p"

def compute_features(cube, dict):
    """Minkowski feature computation with precomputed updates
       Returns V, S, B, X features"""
    cube = np.pad(cube, 1, "constant")
    n_0, n_1, n_2, n_3 = 0, 0, 0, 0
    for x in range(1, cube.shape[0] - 1):
        for y in range(1, cube.shape[1] - 1):
            for z in range(1, cube.shape[-1] - 1):
                dn_3, dn_2, dn_1, dn_0 = dict[
                    hash(tuple(cube[x - 1:x + 2, y - 1:y + 2, z - 1:z + 1].flatten()))]
                n_3 += dn_3;
                n_2 += dn_2;
                n_1 += dn_1;
                n_0 += dn_0;
    V = n_3;
    S = -6 * n_3 + 2 * n_2;
    B = 3 * n_3 / 2 - n_2 + n_1 / 2;
    X = - n_3 + n_2 - n_1 + n_0
    return V, S, B, X

def compute_batch(batch):
    global path
    dict = pickle.load(open(path, "rb"))
    features = list(map(lambda b: compute_features(b, dict), batch))
    return features

def compute_batch_parallel(batch, num_threads=4):
    slice_size = round(len(batch)/num_threads+0.5)
    starts = list(range(0, num_threads*slice_size, slice_size))
    batches = [batch[start:start+slice_size] for start in starts]
    pool = mp.Pool(num_threads)
    result = []
    res = [pool.apply_async(compute_batch, args=(b,), callback=lambda lst: result.extend(lst)) for b in batches]
    pool.close()
    pool.join()
    return result