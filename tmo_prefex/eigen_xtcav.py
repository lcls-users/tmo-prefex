#!/cds/sw/ds/ana/conda2/inst/envs/ps-4.2.5/bin/python3

import numpy as np
import sys
import re
import h5py
from scipy.fftpack import dct
import multiprocessing as mp
from datetime import datetime
import time

'''
data = np.array([images[i,:,:].flatten() for i in range(images.shape[0])])
avgim = np.mean(data,axis=0)
residual = np.array([data[i,:]-avgim for i in range(data.shape[0])])
COVMAT = np.cov(residual.T)
vals,vecs = np.linalg.eig(COVMAT) # this peice takes a long time.

Then remove mean, then remove a handfull of np.dot(eigvecs,data) and plot the >0 pixels
'''

def processBlock(params:dict) -> dict:
    tstart = time.process_time_ns()
    with h5py.File('%s/%s.h5'%(params['inpath'],params['infile']),'r') as inh5:
        (nims,xdim,ydim) = inh5['xtcav']['images'][()].shape 
        nims //= params['nblocks'] 
        print('newshape = (%i,%i,%i)'%(nims,xdim,ydim))
        datablock = np.column_stack([inh5['xtcav']['images'][()][params['tid']+params['nblocks']*j,::params['skipx'],::params['skipy'] ].flatten() for j in range(nims) ] )
        print('datablock %i shape\t%s'%(params['tid'],datablock.shape))
    params['meanimg'] = np.mean(datablock,axis=1)
    print("params['meanimg'].shape",params['meanimg'].shape)
    for i in range(datablock.shape[1]):
        datablock[:,i] -= params['meanimg'].astype(np.int16)
    tfread = time.process_time_ns()
    print('File read time [msec]\t%i'%(int(tfread-tstart)//1000000))
    outfile = '%s.eigenxtcav.tid%i.h5'%(params['infile'],params['tid'])
    params['outfile'] = outfile
    covMat = np.cov(datablock)
    tcov = time.process_time_ns()
    print('covMat time [msec]\t%i'%(int(tcov-tfread)//1000000))
    print('covMat.shape',covMat.shape)
    eigVals,eigMat = np.linalg.eig(covMat)
    teig = time.process_time_ns()
    print('eigMat time [msec]\t%i'%(int(teig-tcov)//1000000))
    print('eigMat.shape',eigMat.shape)
    params['eigvals'] = eigVals.real
    params['eigmat'] = eigMat.real
    return params

def main() -> None:
    current_time = datetime.now().strftime("%H:%M:%S")
    print("started: \t%s"%(current_time))

    if len(sys.argv) < 3:
        print('Syntax: eigen_xtcav.py <h5 full path and name> <outputpath>')
        return
    fullinfile = sys.argv[1]
    m = re.search('^(.+)/(.+)\.h5$',fullinfile)
    if not m:
        print(fullinfile)
        return
    nblocks = 1<<5 
    paramslist = [{} for i in range(nblocks)]
    for i,p in enumerate(paramslist):
        p['inpath'] = str(m.group(1))
        p['infile'] = str(m.group(2))
        p['outpath'] = str(sys.argv[2])
        p['nblocks'] = np.uint8(nblocks)
        p['tid'] = np.uint8(i)
        p['skipx'] = 4
        p['skipy'] = 4

    #pool = mp.Pool(len(paramslist))
    paramslist = mp.Pool(mp.cpu_count()).map(processBlock,paramslist)

    np.savetxt('%s/%s.eigvals.dat'%(paramslist[0]['outpath'],paramslist[0]['infile']),np.column_stack([np.log2(np.abs(p['eigvals'])) for p in paramslist]),fmt='%.6f')
    _= [np.savetxt('%s/%s.eigmat%i.dat'%(p['outpath'],p['infile'],p['tid']),p['eigmat'],fmt='%.6f') for p in paramslist]

    current_time = datetime.now().strftime("%H:%M:%S")
    print("finished: \t%s"%(current_time))

    return

if __name__ == '__main__':
    main()
