#!/usr/bin/python3
import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
import re


def main():
    fname = sys.argv[1]
    with h5py.File(fname,'r') as f:
        detstr = 'mrco_hsd'
        ports = f[detstr].keys()
        portnums = np.sort([int(re.match('(\d+)',k).group(1)) for k in ports])
        for i, p in enumerate(portnums):
            t = f[detstr][str(p)]['tofs'][()]
            tvalid = [int(v)>>1 for v in t if v<(1<<14)]
            #print(p, tvalid)
            #continue
            h = np.zeros((1<<int(np.log2(np.max(tvalid, initial=1)))+1))
            for v in tvalid:
                h[v] += 1
            plt.title('port %i'%p)
            plt.plot(h,label='port %i'%p)
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
