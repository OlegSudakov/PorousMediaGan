from minkowskiMeasures import compute_batch_parallel
import numpy as np
import matplotlib.pyplot as plt
import tifffile

class CoreData:
    def __init__(self, samplesize=(400, 400, 400), filepath="../../data/berea/berea.tif",
                samplename="Berea", num_threads=4):
        self.sample = tifffile.imread(str(filepath))
        self.samplename = samplename
        self.samplesize = samplesize
        self.num_threads = num_threads
        
    def show_slice(self, layer):
        """Displays a slice of a sample matrix"""
        plt.figure(figsize = (6, 6))
        plt.tick_params(axis='both', labelbottom='off',labelleft='off',left='off', bottom='off')
        plt.title("{} slice {}".format(self.samplename, layer), fontsize = 14)
        plt.imshow(self.sample[layer], cmap = "gray")
        
    def next_batch(self, batch_size=128, batch_sample_size=(64, 64, 64)):
        h, w, l = batch_sample_size
        sample_h, sample_w, sample_l = self.samplesize
        batch = []
        for i in range(batch_size):
            h_offset = np.random.choice(np.arange(sample_h - h))
            w_offset = np.random.choice(np.arange(sample_w - w))
            l_offset = np.random.choice(np.arange(sample_l - l))
            batch.append(self.sample[h_offset:h_offset+h, w_offset:w_offset+w, l_offset:l_offset+l])
        batch = [sample/255 for sample in batch]
        minkowski_functionals = compute_batch_parallel(batch, self.num_threads)
        minkowski_functionals = np.array(minkowski_functionals)
        batch = np.array(batch)
        batch = batch.reshape(batch.shape[0], 1, batch.shape[1], batch.shape[2], batch.shape[3])
        return batch, minkowski_functionals