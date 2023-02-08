#!/cds/sw/ds/ana/conda2/inst/envs/ps-4.5.7-py39/bin/python3

import matplotlib.pyplot as plt
import h5py
import numpy as np
import sys
import re


def main(path,fname,runstring):
    print('%s/%s'%(path,fname))
    #path = '/reg/data/ana16/tmo/tmox42619/scratch/ryan_output_vernier_1000vlsthresh/h5files/'
    #fname = 'hits.tmox42619.run_188.h5.counts.qtofs.qgmd.h5'
    gmdinds = [i for i in range(1,32,5)]
    with h5py.File('%s/%s'%(path,fname),'r') as f:
        portkeys = [k for k in f.keys() if (re.search('port',k) and not re.search('_16',k) and not re.search('_2',k))] # keeping the bare MCP ports 2 and 16 here
        gmdscale = 0.5*(f['gmd']['bins'][()][:-1]+f['gmd']['bins'][()][1:])/f['gmd']['norm'][()]
        '''
        for k in portkeys:
            _= [plt.plot(f[k]['quantbins'][()][:-1]/8./6.,np.cumsum(f[k]['hist'][()][i,:]*gmdscale[i])) for i in gmdinds]
            plt.xlabel('tof [ns]')
            plt.ylabel('mean cumulative counts/shot')
            plt.ylim((0,25))
            plt.xlim((690,1300))
            plt.title('%s %s'%(runstring,k))
            plt.grid()
            plt.legend(['%i uJ'%int(0.5*(f['gmd']['bins'][()][i]+f['gmd']['bins'][()][i+1])+0.5) for i in gmdinds],bbox_to_anchor=(1.05,1),loc='upper left')
            plt.tight_layout()
            plt.savefig('/cds/home/c/coffee/Downloads/%s_cumcounts_%s.png'%(runstring,k))
        '''
        for k in portkeys:
            _= [plt.plot(np.cumsum(f[k]['hist'][()][i,:]*gmdscale[i])) for i in gmdinds]
            plt.title('%s %s'%(runstring,k))
            plt.xlabel('quantized bins')
            plt.ylabel('mean cumulative counts/shot')
            plt.legend(['%i uJ'%int(0.5*(f['gmd']['bins'][()][i]+f['gmd']['bins'][()][i+1])+0.5) for i in gmdinds],bbox_to_anchor=(1.05,1),loc='upper left')
            plt.tight_layout()
            plt.grid()
            plt.savefig('/cds/home/c/coffee/Downloads/%s_cumcounts_%s_qbins.png'%(runstring,k))
            plt.show()

if __name__ == '__main__':
    if len(sys.argv)>1:
        m = re.search('(.*/h5files)/(hits.*(run_\d+)\.h5\.counts\..*\.h5)',sys.argv[1])
        if m:
            main(m.group(1),m.group(2),m.group(3))
        else:
            print('failed the path/fname match')
    else:
        print('syntax: figs/plotting.counts.py fnameQuantizedHist')
