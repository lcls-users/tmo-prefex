#!/usr/bin/python3
import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
import re


def main():
    fname = sys.argv[1]
    hist = {}
    maxtof:np.uint32 = 1<<15
    scramble = []
    with h5py.File(fname,'r') as f:
        runstr = [k for k in f.keys()][0]
        detstr = 'mrco_hsd'
        hsd = f[runstr][detstr]
        ports = [p for p in hsd.keys()]
        #scramble = [p for p in hsd.keys()]
        #np.random.shuffle(scramble)
        portnums = np.sort([int(re.search('_(\d+)$',k).group(1)) for k in ports])
        #portnums = [int(re.search('_(\d+)$',k).group(1)) for k in scramble]
        print("portnums:\t",portnums)
        for i,p in enumerate(ports):
            t = hsd[p]['tofs'][()]
            tvalid = [v for v in t if v<maxtof]
            hist.update({p:[0]*maxtof})
            for v in tvalid:
                hist[p][v] += 1

    fig,axs = plt.subplots(nrows=4, ncols=4, figsize=(25, 24))
    for i,p in enumerate(hist.keys()):
    #for i,p in enumerate(scramble):
        col = i%4
        row = i>>2
        axs[row,col].stairs(hist[p][11000:15000],label=p)
        axs[row,col].legend()
        axs[row,col].set_xlabel('tof bins * expand')
        axs[row,col].set_ylabel('counts')
    plt.show()


if __name__ == "__main__":
    if len(sys.argv)>1:
        main()
    else:
        print('point me to an .h5 file!')

