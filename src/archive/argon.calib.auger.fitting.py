#!/usr/bin/python3

import h5py
import numpy as np
import sys
import re

def fit(xin,yin):
    params = {}
    x = np.log2(xin)
    y = np.log2(yin)
    xm = np.mean(x)
    ym = np.mean(y)
    s = 2**8
    params['log2'] = True
    params['xmean'] = xm
    params['ymean'] = ym
    params['scale'] = s
    X = np.stack((np.ones(x.shape),s*(x-xm),np.power(s*(x-xm),int(2))),axis=-1) 
    params['theta'] = np.dot(s*(y-ym),np.linalg.pinv(X.T))
    return params

def transform(xin,params):
    x = xin
    if params['log2']:
        x = np.log2(x)-params['xmean']
    else:
        x -= params['xmean']
    x *= params['scale']
    X = np.stack((np.ones(x.shape),x,np.power(x,int(2))),axis=-1)
    y = np.dot(X,parmas['theta'])
    yout = y/params['scale'] + params['ymean']
    if params['log2']:
        return 2**yout
    return yout

def main():
    if len(sys.argv)<2:
        print('give me the argon.auger.calib h5 calib file to fit')
        return
    m = re.search('^(.*)/((.*)\.h5)',sys.argv[1])
    if not m:
        print('failed filename matching')
        return
    print(m.group(1),m.group(2),m.group(3))
    path = m.group(1)
    filehead = m.group(3)

    with h5py.File('%s/%s.calibfit.h5'%(path,filehead),'w') as fitfile:
        with h5py.File(sys.argv[1],'r') as f:
            for p in f.keys():
                if p in fitfile:
                    pgrp = fitfile[p]
                else:
                    pgrp = fitfile.create_group(p)
                for r in f[p].keys():
                    run = f[p][r]['tofs'].attrs['run']
                    ret = f[p][r]['tofs'].attrs['ret']
                    toffset = f[p][r]['tofs'].attrs['toffset']

                    if '%s'%ret in pgrp:
                        retgrp = pgrp['%s'%ret]
                    else:
                        retgrp = pgrp.create_group('%s'%ret)

                    phEn = f[p][r]['tofs'].attrs['phEn']
                    if '%s'%phEn in retgrp:
                        phgrp = retgrp['%s'%phEn]
                    else:
                        phgrp = retgrp.create_group('%s'%phEn)

                    params = fit(f[p][r]['tofs'][()], f[p][r]['energies'][()])

                    if 'theta' in phgrp:
                        i = 0
                        while 'theta_%i'%i in phgrp:
                            i += 1
                        theta = phgrp.create_dataset('theta_%i'%i,data=params['theta'])
                    else:
                        theta = phgrp.create_dataset('theta',data=params['theta'])

                    theta.attrs.create('xmean',params['xmean'])
                    theta.attrs.create('ymean',params['ymean'])
                    theta.attrs.create('scale',params['scale'])
                    theta.attrs.create('log2',params['log2'])
                    theta.attrs.create('run',run)
                    theta.attrs.create('toffset',toffset)

                    print(params)

    return

if __name__ == '__main__':
    main()

