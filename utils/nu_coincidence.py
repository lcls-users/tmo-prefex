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
    hist = np.zeros((1<<10,1<<10),dtype=np.uint64)
    #hist = np.zeros((1<<10,1<<10),dtype=np.uint64)
    h = {}
    tofs = {}
    addresses = {}
    nedges = {}
    maxtof:np.uint32 = 1<<15
    t0=11750
    scale = 1
    for name in filenames:
        print(name)
        with h5py.File(os.path.join(path,name),'r') as f:
            runstr = [k for k in f.keys()][0]
            detstr = 'mrco_hsd'
            hsd = f[runstr][detstr]
            fzp = f[runstr]['tmo_fzppiranha']
            xgmd = f[runstr]['xgmd']
            ports = [p for p in hsd.keys()]
            ens = xgmd['energy'][()]
            cents = fzp['centroids'][()]
        
            corrports = ['port_000','port_180']

            h.update({corrports[0]:[int(0)]*(hist.shape[0])})
            h.update({corrports[1]:[int(0)]*(hist.shape[1])})

            for p in corrports:
                tofs.update({p:hsd[p]['tofs'][()]})
                addresses.update({p:hsd[p]['addresses'][()]})
                nedges.update({p:hsd[p]['nedges'][()]})
                for t in tofs[p]:
                    h[p][ max(0,min(len(h[p])-1, int(t-t0)))] +=1
            scale = len(addresses[corrports[0]])
            for i in range(len(addresses[corrports[0]])):
                n1 = nedges[corrports[0]][i]
                n2 = nedges[corrports[1]][i]
                a1 = addresses[corrports[0]][i]
                a2 = addresses[corrports[1]][i]
                for t1 in tofs[corrports[0]][a1:a1+n1]:
                    for t2 in tofs[corrports[1]][a2:a2+n2]:
                        ind1 = max(0,min(hist.shape[0]-1, int(t1-t0)))
                        ind2 = max(0,min(hist.shape[1]-1, int(t2-t0)))
                        hist[ind1,ind2] += 1
            #falsecoince = np.outer(h[corrports[0]],h[corrports[1]]).astype(float)
            #falsecoince /= float(scale>>4)


    #plt.imshow(falsecoince,clim=(0,10))
    #plt.colorbar()
    #plt.show()
    plt.imshow(hist,clim=(0,25))
    plt.colorbar()
    plt.show()

    '''
    fig,axs = plt.subplots(nrows=4, ncols=4, figsize=(25, 24))
    for i,p in enumerate(hist.keys()):
        col = i%4
        row = i>>2
        axs[row,col].imshow(hist[p]) 
        axs[row,col].set_title(str(np.sum(hist[p]))) 
    plt.show()
    '''


if __name__ == "__main__":
    if len(sys.argv)>1:
        main()
    else:
        print('point me to an .h5 file!')


