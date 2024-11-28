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
    maxtof:np.uint32 = 1<<16
    runstr = 'r0273'
    m = re.search('\.(r\d+)\.',filenames[0])
    if m:
        runstr = m.group(1)
    for name in filenames:
        print(name)
        with h5py.File(os.path.join(path,name),'r') as f:
            detstr = 'mrco_hsd'
            hsd = f[detstr]
            fzp = f['tmo_fzppiranha']
            xgmd = f['xgmd']
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
    startinds = [(v<<1)-(1<<10) for v in [24000, 24150, 24150, 24100, 24090, 23950, 23950, 24150, 24150, 24150, 25250, 24100, 24300, 24150, 24000, 24000 ]]
    window = [5000]*len(startinds)
    for i,p in enumerate(hist.keys()):
        col = i%4
        row = i>>2
        axs[row,col].stairs(hist[p][startinds[i]:startinds[i]+window[i]],label=p)
        axs[row,col].legend()
        axs[row,col].set_xlabel('tof bins * inflate * expand')
        axs[row,col].set_ylabel('counts')
    plt.savefig('./figures/ArgonAugers_%s.png'%runstr)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv)>1:
        main()
    else:
        print('point me to the run folder, the one with all the chunks!')


