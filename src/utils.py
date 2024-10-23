#!/usr/bin/python3

import numpy as np
import h5py
import math
import os
import re
from scipy.fftpack import dct,dst

def show(x):
    offset = 5+np.abs(np.min(x))
    print(' '*(offset-1)+'|')
    _ = [print(' '*int(offset+v-1)+'*') for v in x]
    print(' '*(offset-1)+'|')
    return

def dctmat(sz):
    return 2*dct( np.identity(2*sz),axis=1 , type=2)[::2].T
def dstmat(sz):
    return 2*dst( np.identity(2*sz),axis=1 , type=2)[1::2].T
def idctmat(sz):
    return 1./2*dct( np.identity(2*sz),axis=0 , type=3)[::2].T
def idstmat(sz):
    return 1./2*dst( np.identity(2*sz),axis=0 , type=3)[1::2].T

def mydct(cmat,x):
    xin = np.append(x[::2],x[-1::-2])
    return np.inner(cmat,xin)[::2]

def mydst(smat,x):
    xin = np.append(x[1::2],-1*x[-2::-2])
    return np.inner(smat,xin)[1::2]

# then add the frequency scaled version
# then add the zeros append for oversampling
def quick_mean(lints,baseshift=2):
    if len(lints)<(1<<baseshift):
        print('too short for quick mean')
        return None
    return np.uint32( np.sum(lints[:1<<baseshift])>>baseshift )



def randomround(x:float,rng):
    return (np.int64(x) + np.int64(x%1>rng.random()))

def inlims(x,low,high):
    if x<low:
        return False
    if x>=high:
        return False
    return True

def checkdet(runlist,detname):
    for r in runlist:
        if not detname in r.detnames:
            return False 
    return True

def PWRspectrum(wv):
    return np.power(abs(np.fft.fft(wv).real),int(2))

def rollon(vec,n):
    vec[:int(n)] = vec[:int(n)]*np.arange(int(n),dtype=float)/float(n)
    return vec

def tanhFloat(x,thresh=1):
    y = 0.5*(1.0 + np.tanh(x.astype(float)/thresh))
    return y.astype(type(x[0]))

def tanhInt(x,bits=8):
    y = (1<<(bits-1))*(1+np.tanh(x.astype(float)))
    return y.astype(type(x[0]))

def pkey(p):
    return 'port_%i'%p

def mypoly(x,order=4):
	result = np.ones((x.shape[0],order+1),dtype=float)
	result[:,1] = x.copy()
	if order < 2:
		return result
	for p in range(2,order+1):
		result[:,p] = np.power(result[:,1],int(p))
	return result

def fitpoly(x,y,order=4):
    assert len(x)==len(y)
    assert len(x)>order
    x0 = np.mean(np.array(x))
    theta = np.linalg.pinv( mypoly(np.array(x-x0).astype(float),order=order) ).dot(np.array(y).astype(float)) # fit a polynomial (order 3) to the points
    return x0,theta

def fitval(x,theta):
    y = float(0.)
    for p,th in enumerate(theta):
        y += float(th)*math.pow(x,int(p))
    return y

def fitcurve(x,theta):
    y = np.zeros(x.shape,dtype=float)
    for p in range(theta.shape[0]):
        y += theta[p]*np.power(x,int(p))
    return y

def getcentroid(inds,spec):
    x = inds
    y = spec 
    num = np.sum(x*y)
    denom = np.sum(y)
    if (denom>0):
        return int(num/denom)
    return 0

def makehist(fname,bins=(np.arange(2**10+1)-2**9)*2**10):
    with h5py.File(fname,'r') as f:
        for p in f.keys():
            data = []
            for k in f[p]['waves'].keys():
                data += list(f['port_0']['waves'][k][()])
            h,b = np.histogram(data,bins)
            np.savetxt('histsig.%s.dat'%p,np.column_stack((b[:-1],h)),fmt='%i')
    return

def samples2ascii(ipath,fname,opath):
    logics={}
    waves={}
    with h5py.File('%s/%s'%(ipath,fname),'r') as f:
        for p in f.keys(): 
            logics[p] = np.stack([f[p]['logics'][k][()] for k in f[p]['logics'].keys()],axis=-1)
        for p in f.keys(): 
            waves[p] = np.stack([f[p]['waves'][k][()] for k in f[p]['waves'].keys()],axis=-1)
        _=[np.savetxt('%s/%s.logics.%s.dat'%(opath,fname,p),logics[p],fmt='%i') for p in logics.keys()]
        _=[np.savetxt('%s/%s.waves.%s.dat'%(opath,fname,p),waves[p],fmt='%i') for p in waves.keys()]
    return

def fillconfigs(cfgname):
    print(cfgname)
    print('%s/%s.%s'%(os.getenv('scratchpath'),os.getenv('expname'),'hsdconfigs.h5'))
    params = {'chans':{},'t0s':{},'logicthresh':{}}
    with h5py.File(cfgname,'r') as f:
        params['inflate'] = f.attrs['inflate']
        params['expand'] = f.attrs['expand']
        params['vlsthresh'] = f.attrs['vlsthresh']
        params['vlswin'] = f.attrs['vlswin']
        params['l3offset'] = f.attrs['l3offset']
        for p in f.keys():
            m = re.search('^\w+_(\d+)$',p)
            if m:
                k = int(m.group(1))
                params['chans'][k] = f[p].attrs['hsd']
                params['t0s'][k] = f[p].attrs['t0']
                params['logicthresh'][k] = f[p].attrs['logicthresh']
    return params

def xtcav_crop(inimg,win=(256,256)):
    # hard coded factor of 2 scale down
    xprof = np.mean(inimg,axis=0)
    yprof = np.mean(inimg,axis=1)
    y0 = np.argmax(xprof)
    x0 = np.argmax(yprof)
    resimg = (np.roll(inimg,(-x0+win[0]//2,-y0+win[1]//2),axis=(0,1)))[:win[0],:win[1]]
    tmp= np.column_stack((resimg,np.flip(resimg,axis=1)))
    outimg=np.row_stack((tmp,np.flip(tmp,axis=0)))
    W = dct(dct(outimg,axis=1,type=2),axis=0,type=2)
    xenv = np.zeros(W.shape[0])
    yenv = np.zeros(W.shape[1])
    xenv[:win[0]//2] = 0.5*(1+np.cos(np.arange(win[0]//2)*np.pi/(win[0]/2)))
    yenv[:win[1]//2] = 0.5*(1+np.cos(np.arange(win[1]//2)*np.pi/(win[1]/2)))
    for i in range(W.shape[1]//2):
        W[:,i] *= xenv
    for i in range(W.shape[0]//2):
        W[i,:] *= yenv
    W *= 4.0/np.product(W.shape)
    out = dct( dct(W[:win[0]//2,:win[1]//2],type=3,axis=0),type=3,axis=1)[:win[0]//4,:win[1]//4]
    return out,x0//2,y0//2
    #return dct(dct(W,axis=2,type=3),axis=1,type=3),x0,y0
    #print(x0,y0)
    #return inimg[:win[0],:win[1]],x0,y0
