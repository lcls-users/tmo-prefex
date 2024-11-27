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
    filenames = np.sort([nm for nm in os.listdir(path) if os.path.isfile(os.path.join(path, nm))])
    corrports = ['port_090','port_270','port_000','port_180']
    #filenames = [nm for nm in os.listdir(path) if os.path.isfile(os.path.join(path, nm))]

    hist = [np.zeros((1<<9,1<<9),dtype=np.uint64)]*len(corrports)
    #hist = np.zeros((1<<10,1<<10),dtype=np.uint64)
    h = {}
    c = {}
    tofs = {}
    addresses = {}
    nedges = {}
    maxtof:np.uint32 = 1<<15
    t0=11000
    scale = 1

    fig,axs=plt.subplots(1,4,figsize=(24,12))

    for k,p in enumerate(corrports):
        for nameind,name in enumerate(filenames):
            with h5py.File(os.path.join(path,name),'r') as f:
                runstr = [k for k in f.keys()][0]
                detstr = 'mrco_hsd'
                hsd = f[runstr][detstr]
                fzp = f[runstr]['tmo_fzppiranha']
                xgmd = f[runstr]['xgmd']
                ports = [p for p in hsd.keys()]
                ens = xgmd['energy'][()]
                cents = fzp['centroids'][()]
        
    
                if p not in h.keys():
                    h.update({p:[int(0)]*(hist[k].shape[0])})
                    c.update({'fzp':[int(0)]*(hist[k].shape[1])})

                tofs.update({p:hsd[p]['tofs'][()]})
                addresses.update({p:hsd[p]['addresses'][()]})
                nedges.update({p:hsd[p]['nedges'][()]})
                for t in tofs[p]:
                    h[p][ max(0,min(len(h[p])-1, int(t-t0)))] +=1
                for i in range(len(addresses[p])):
                    n1 = nedges[p][i]
                    a1 = addresses[p][i]
                    for t1 in tofs[p][a1:a1+n1]:
                        ind1 = max(0,min(hist[k].shape[0]-1, int(t1-t0)>>4))
                        ind2 = max(0,min(hist[k].shape[1]-1, int(cents[i])>>2))
                        hist[k][ind1,ind2] += 1

            axs[k].imshow(hist[k],clim=(0,10))
            axs[k].set_xlabel('fzp')
            axs[k].set_ylabel('hsd')
            axs[k].set_title('%s'%(p))

    #plt.imshow(falsecoince,clim=(0,10))
    #plt.show()
    #plt.colorbar()
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

