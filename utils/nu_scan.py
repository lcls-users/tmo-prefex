#!/usr/bin/python3
import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
import re
import math
import statistics as stats

def main():
    fname = sys.argv[1]
    hist = {}
    maxtof:np.uint32 = 1<<14
    scramble = []
    with h5py.File(fname,'r') as f:
        runstr = [k for k in f.keys()][0]
        detstr = 'mrco_hsd'
        hsd = f[runstr][detstr]
        fzp = f[runstr]['tmo_fzppiranha']
        xgmd = f[runstr]['xgmd']
        ports = [p for p in hsd.keys()]
        portnums = np.sort([int(re.search('_(\d+)$',k).group(1)) for k in ports])
        print("portnums:\t",portnums)
        ens = xgmd['energy'][()]
        cents = fzp['centroids'][()]
        plt.plot(cents,'+')
        plt.show()
        '''
        h = [0]*(1<<11)
        for v in cents:
            if v<len(h):
                h[v] += 1
        plt.stairs(h)
        plt.show()
        '''
        
        for p in ports[::4]:
            print(p)
            tofs = hsd[p]['tofs'][()]
            addresses = hsd[p]['addresses'][()]
            nedges = hsd[p]['nedges'][()]
            hist.update({p:np.zeros(shape=(1<<8,1<<8),dtype=np.uint32)}) # [[0]*maxtof for c in range(1<<8)]})
            print(hist[p].shape)
            for i,a in enumerate(addresses):
                n = nedges[i]
                c = min(cents[i]>>3,(1<<8)-1)
                tvalid = [int(v-(1<<13))>>5 for v in tofs[a:a+n] if (int(v-(1<<13))>>5)<(1<<8)]
                for v in tvalid:
                    hist[p][v,c] += 1

    fig,axs = plt.subplots(nrows=4, ncols=4, figsize=(25, 24))
    for i,p in enumerate(hist.keys()):
        col = i%4
        row = i>>2
        axs[row,col].imshow(hist[p])
    plt.show()


if __name__ == "__main__":
    if len(sys.argv)>1:
        main()
    else:
        print('point me to an .h5 file!')


