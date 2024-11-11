#!/usr/bin/python3
import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
import re


def main():
    fname = sys.argv[1]
    hist = {}
    xgmdsum = 0
    maxtof:np.uint32 = 1<<15
    with h5py.File(fname,'r') as f:
        runstr = [k for k in f.keys()][0]
        hsd = f[runstr]['mrco_hsd']
        xgmd = [min(e>>3,1<<8) for e in f[runstr]['xgmd']['energy']]
        normlist = [0]*(1<<8)

        ports = [p for p in hsd.keys()]
        #portnums = np.sort([int(re.search('_(\d+)$',k).group(1)) for k in ports])
        portnums = [int(re.search('_(\d+)$',k).group(1)) for k in ports]
        print("portnums:\t",portnums)
        for p in ports:
            hist.update({p:[[0]*maxtof]})

            tofs = hsd[p]['tofs'][()]
            adrs = hsd[p]['addresses'][()]
            nedg = hsd[p]['nedges'][()]
            for i,a in adrs[]:
                tvalid = [v for v in t[a:a+nedg[i]] if v<maxtof]
                for v in tvalid:
                    hist[p][v] += 1

    fig,axs = plt.subplots(nrows=4, ncols=4, figsize=(25, 24))
    for i,p in enumerate(hist.keys()):
        col = i%4
        row = i>>2
        axs[row,col].stairs(hist[p][10000:15000],label=p)
        axs[row,col].legend()
        axs[row,col].set_xlabel('tof bins * expand')
        axs[row,col].set_ylabel('counts')
    plt.show()


if __name__ == "__main__":
    if len(sys.argv)>1:
        main()
    else:
        print('point me to an .h5 file!')


