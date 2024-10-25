#!/usr/bin/python3
import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
import re


def main():
    fname = sys.argv[1]
    with h5py.File(fname,'r') as f:
        runstr = [k for k in f.keys()][0]
        detstr = 'mrco_hsd'
        ports = f[runstr][detstr].keys()
        portnums = np.sort([int(re.search('_(\d+)$',k).group(1)) for k in ports])
        for p in portnums:
            t = f[runstr][detstr]['port_%i'%p]['tofs'][()]
            tvalid = [int(v)>>1 for v in t if v<(1<<14)]
            h=np.zeros((1<<int(np.log2(np.max(tvalid)))+1))
            for v in tvalid:
                h[v] += 1
            plt.title('port_%i'%p)
            plt.plot(h,label='port_%i'%p)
            plt.legend()
            plt.xlim(2800,3500)
            plt.xlabel('histbins = 2 hsd bins')
            plt.ylabel('hits')
            plt.show()

if __name__ == "__main__":
    if len(sys.argv)>1:
        main()
    else:
        print('point me to an .h5 file!')
