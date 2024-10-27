#!/cds/sw/ds/ana/conda2/inst/envs/ps-4.2.5/bin/python3

import h5py
import matplotlib.pyplot as plt
import re
import numpy as np


qmeans = {}
qdiffs = {}

qname = '/reg/data/ana16/tmo/tmox42619/scratch/ryan_output_multiretardation/h5files/quantHist_nonuniform_runs132-133-134.NNO.qknob0.000.tmox42619.h5'
with h5py.File(qname,'r') as q:
    portkeys = [k for k in q.keys() if re.search('port',k)]
    for p in portkeys:
        qmeans[p] = (q[p]['qbins'][()][:-1]+q[p]['qbins'][()][1:])*0.5
        qdiffs[p] = q[p]['qbins'][()][1:]-q[p]['qbins'][()][:-1]
        #plt.plot(qmeans[p],1./qdiffs[p],'.',label=p)
    '''
    plt.legend()
    plt.show()
    '''
    p = 'port_12'
    plt.subplot(2,2,1)
    plt.pcolor(-1*q[p]['hist'][()][:,::(1<<3)].T,vmin=-2,vmax=0,cmap='gray')
    plt.xlabel('bins')
    plt.ylabel('sample stream of images')
    plt.title('horiz')

    plt.subplot(2,2,2)
    plt.stairs(1./qdiffs[p])
    plt.ylabel('counts/bin widths')
    plt.xlabel('bins')
    plt.title('horiz')

    p = 'port_0'
    plt.subplot(2,2,3)
    plt.pcolor(-1*q[p]['hist'][()][:,::(1<<3)].T,vmin=-2,vmax=0,cmap='gray')
    plt.ylabel('sample stream of images')
    plt.xlabel('bins')
    plt.title('vert')

    plt.subplot(2,2,4)
    plt.stairs(1./qdiffs[p])
    plt.ylabel('counts/bin widths')
    plt.xlabel('bins')
    plt.title('vert')
    plt.savefig('./figures/SUmmitPlusCompare.png')
    plt.show()


