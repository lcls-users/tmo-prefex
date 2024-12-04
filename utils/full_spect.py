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
    maxtof:np.uint32 = 1<<17
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
    #startinds = [(v<<1)-(1<<10) for v in [24000, 24150, 24150, 24100, 24090, 23950, 23950, 24150, 24150, 24150, 25250, 24100, 24300, 24150, 24000, 24000 ]]
    #startinds = [3650, 4250, 4500, 4000, 4000, 3500, 3500, 4500, 4500, 4250, 8750, 4250, 4750, 4500, 3750, 3750]
    #startinds = [1650, 2250, 2500, 2000, 2000, 1500, 1500, 2500, 2500, 2250, 6750, 2250, 2750, 2500, 1750, 1750]
    #
    # removing another 2000 for sake of _longAugers_new.h5
    startinds = [-350, 250, 500, 000, 000, -500, -500, 500, 500, 250, 4750, 250, 750, 500, -250, -250]
    for i in range(len(startinds)):
        startinds[i] += (1<<16)+30000
        
    window = [14000]*len(startinds)
    for i,p in enumerate(hist.keys()):
        col = i%4
        row = i>>2
        axs[row,col].stairs(hist[p][startinds[i]:startinds[i]+window[i]],label='%s\n%i'%(p,startinds[i]))
        axs[row,col].legend()
        axs[row,col].set_xlabel('tof bins * inflate * expand')
        axs[row,col].set_ylabel('counts')
    plt.savefig('./figures/ArgonAugers_%s.png'%runstr)
    plt.show()

    outpath = sys.argv[2]
    os.makedirs(outpath,mode=0o776,exist_ok=True)
    nm = '%s_full_spect.h5'%(runstr)
    oname = os.path.join(outpath,nm)
    with h5py.File(oname,'w') as o:
        for i,p in enumerate(hist.keys()):
            dset = o.create_dataset(p,data=hist[p][startinds[i]:startinds[i]+window[i]],dtype=np.uint16)
            dset.attrs.create('startind',data = startinds[i])
            dset.attrs.create('window',data = window[i])



if __name__ == "__main__":
    if len(sys.argv)>1:
        main()
    else:
        print('point me to the run folder, the one with all the chunks!')


