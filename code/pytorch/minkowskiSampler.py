import numpy as np
import pickle

class MinkowskiSampler:
    def __init__(self, dictpath = "../../data/dicts/berea_mm_dist.p"):
        self.dict = pickle.load(open(dictpath, "rb"))

    def normal_to_actual(self, val):
        for i in range(4):
            mu, std = self.dict[i]
            val[:, i] = val[:, i] * std + mu
        return val*std + mu
        
    def actual_to_normal(self, val):
        for i in range(4):
            mu, std = self.dict[i]
            val[:, i] = (val[:, i] - mu) / std
        return val

    def sample_n(self, batch_size):
        vals = np.zeros((batch_size, 4))
        for i in range(4):
            mu, std = self.dict[i]
            vals[:, i] = np.random.normal(loc=mu, scale=std, size=batch_size)
        return vals

    def fill_cube(self, mm, imageSize):
        cube = np.zeros((mm.shape[0], mm.shape[1], imageSize, imageSize, imageSize))
        for i in range(mm.shape[0]):
            for j in range(mm.shape[1]):
                cube[i, j, :, :, :] = mm[i, j]
        return cube