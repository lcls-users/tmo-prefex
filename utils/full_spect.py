#!/usr/bin/python3
import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
import re
import os


def main():
    path = sys.argv[1]
    filenames = [nm for nm in os.listdir(path) if os.path.isfile(os.path.join(path, nm))]
    hist = {}
    maxtof:np.uint32 = 1<<15
    for name in filenames:
        print(name)
        with h5py.File(os.path.join(path,name),'r') as f:
            runstr = [k for k in f.keys()][0]
            detstr = 'mrco_hsd'
            hsd = f[runstr][detstr]
            fzp = f[runstr]['tmo_fzppiranha']
            xgmd = f[runstr]['xgmd']
            ports = [p for p in hsd.keys()]
            portnums = np.sort([int(re.search('_(\d+)$',k).group(1)) for k in ports])
            for i,p in enumerate(ports):
                if p not in hist.keys():
                    hist.update({p:[0]*maxtof})
                tofs = hsd[p]['tofs'][()]
                addresses = hsd[p]['addresses'][()]
                nedges = hsd[p]['nedges'][()]
                t = [max(0,min(maxtof-1,v)) for v in hsd[p]['tofs'][()]]
                for i,a in enumerate(addresses):
                    n = nedges[i]
                    #c = min(max(0,((cents[i]-512)>>4)),(1<<6)-1)
                    t = [max(0,min(len(hist[p])-1, int(v))) for v in tofs[a:a+n]]
                    for v in t:
                        hist[p][v] += 1

    fig,axs = plt.subplots(nrows=4, ncols=4, figsize=(25, 24))
    for i,p in enumerate(hist.keys()):
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
        print('point me to the run folder, the one with all the chunks!')


