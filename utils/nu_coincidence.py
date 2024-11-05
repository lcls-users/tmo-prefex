#!/usr/bin/python3
import sys
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import re
import math
import statistics as stats

def main():
    path = sys.argv[1]
    filenames = [nm for nm in os.listdir(path) if os.path.isfile(os.path.join(path, nm))]
    hist = np.zeros((1<<8,1<<8),dtype=np.uint64)
    tofs = {}
    addresses = {}
    nedges = {}
    maxtof:np.uint32 = 1<<15
    t0=11500
    for name in filenames:
        with h5py.File(os.path.join(path,name),'r') as f:
            runstr = [k for k in f.keys()][0]
            detstr = 'mrco_hsd'
            hsd = f[runstr][detstr]
            fzp = f[runstr]['tmo_fzppiranha']
            xgmd = f[runstr]['xgmd']
            ports = [p for p in hsd.keys()]
            ens = xgmd['energy'][()]
            cents = fzp['centroids'][()]
        
            for p in ['port_000','port_180']:
                tofs.update({p:hsd['port_180']['tofs'][()]})
                addresses.update({p:hsd[p]['addresses'][()]})
                nedges.update({p:hsd[p]['nedges'][()]})
            for i,a1 in enumerate(addresses['port_000']):
                n1 = nedges['port_000'][i]
                n2 = nedges['port_180'][i]
                a2 = addresses['port_180'][i]
                for t1 in tofs['port_000'][a1:a1+n1]:
                    for t2 in tofs['port_180'][a2:a2+n2]:
                        ind1 = max(0,min(hist[p].shape[0]-1, int(t1-t0)>>5))
                        ind2 = max(0,min(hist[p].shape[0]-1, int(t2-t0)>>5)
                        h[ind1,ind2] += 1


    fig,axs = plt.subplots(nrows=4, ncols=4, figsize=(25, 24))
    for i,p in enumerate(hist.keys()):
        col = i%4
        row = i>>2
        axs[row,col].imshow(hist[p]) 
        axs[row,col].set_title(str(np.sum(hist[p]))) 
    plt.show()


if __name__ == "__main__":
    if len(sys.argv)>1:
        main()
    else:
        print('point me to an .h5 file!')


