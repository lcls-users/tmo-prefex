#!/cds/sw/ds/ana/conda2/inst/envs/ps-4.2.5/bin/python3

import h5py
import numpy as np
import sys

def buildweiner(xx,sigmeans,noisemeans):
    W = []
    for i in range(len(sigmeans)):
        nn = np.ones(xx.shape[0],dtype=np.float16)*noisemeans[i]
        yy=-2.*(xx.astype(np.float16)-7)+sigmeans[i]
        W += [np.concatenate((1./(1+np.power(2.,nn-yy)),np.flip(1./(1+np.power(2.,nn-yy)),axis=0)))]
    return np.array(W)

def main():
    baselim = 2000
    path = '/reg/data/ana16/tmo/tmox42619/scratch/ryan_output_multicolorhack/h5files/'
    fname = 'hits.tmox42619.runs_082-083-084-085-086-087-088-089-090-094.h5'
    f = h5py.File('%s%s'%(path,fname),'r')
    data = f['vls']['data'][()]
    spec = np.array([data[i,:]-np.mean(data[i,baselim:]) for i in range(data.shape[0])])
    SPEC = np.fft.fft(spec,axis=1)
    PWR = np.log2(np.abs(SPEC))
    slow,shigh = ((1<<7)-(1<<3),(1<<7)+(1<<3))
    nlow,nhigh = ((1<<9)-(1<<3),(1<<9)+(1<<3))
    nmeans = np.mean(PWR[:,nlow:nhigh],axis=1)
    smeans = np.mean(PWR[:,slow:shigh],axis=1)
    xvec = np.log2(np.arange(1,PWR.shape[1]//2+1))
    Wmat = buildweiner(xvec,smeans,nmeans)
    print(Wmat.shape)
    print(np.max(Wmat))
    print(np.min(Wmat))
    result = np.fft.ifft(Wmat*SPEC,axis=1).real
    np.savez('temp.npz',data=result)
    return 

if __name__ == '__main__':
    main()
