#!/usr/bin/python3
import sys
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import re
import math
import statistics as stats

def quick_centroids(path):
    filenames = [nm for nm in os.listdir(path) if os.path.isfile(os.path.join(path, nm))]
    h = [0]*(1<<11)
    rkey = None
    for name in filenames:
        with h5py.File(os.path.join(path,name),'r') as f:
            if rkey==None:
                rkey=[k for k in f.keys()][0]
            inds = f[rkey]['tmo_fzppiranha']['centroids'][()]
            for v in inds:
                if v<len(h):
                    h[v] +=1
            #print('worked: %s'%name)
    plt.stairs(h)
    plt.xlabel('centroid')
    plt.ylabel('shots')
    plt.show()

    return


def main():
    path = sys.argv[1]
    filenames = [nm for nm in os.listdir(path) if os.path.isfile(os.path.join(path, nm))]
    quick_centroids(path)
    hist = {}
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
            portnums = np.sort([int(re.search('_(\d+)$',k).group(1)) for k in ports])
            #print("portnums:\t",portnums)
            ens = xgmd['energy'][()]
            cents = fzp['centroids'][()]
            '''
            plt.plot(cents,'+')
            plt.show()
            h = [0]*(1<<11)
            for v in cents:
                if v<len(h):
                    h[v] += 1
            plt.stairs(h)
            plt.show()
            '''
        
            for p in ports:
                tofs = hsd[p]['tofs'][()]
                addresses = hsd[p]['addresses'][()]
                nedges = hsd[p]['nedges'][()]
                if p not in hist.keys():
                    hist.update({p:np.zeros(shape=(1<<6,1<<6),dtype=np.uint64)}) # [[0]*maxtof for c in range(1<<8)]})
                for i,a in enumerate(addresses):
                    n = nedges[i]
                    c = min(max(0,((cents[i]-512)>>4)),(1<<6)-1)
                    tvalid = [max(0,min(hist[p].shape[0]-1, int(v-t0)>>5)) for v in tofs[a:a+n]]
                    for v in tvalid:
                        hist[p][v,c] += 1

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


