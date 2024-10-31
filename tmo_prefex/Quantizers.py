#!/cds/sw/ds/ana/conda2/inst/envs/ps-4.2.5/bin/python3

import numpy as np
import h5py
import sys
import re
import math
from utils import fitpoly,fitcurve,fitval
import matplotlib.pyplot as plt
from typing import List
from sklearn.cluster import KMeans

plotting = False

class Quantizer:
    def __init__(self,style='nonuniform',nbins=1024):
        self.style:str = style
        self.wave:bool = False
        if style == 'wave':
            self.wave = True
        self.nbins:np.uint32 = nbins
        self.qbins:List[float] = []

    def setbins(self,data,knob=0.0):
        if self.style=='nonuniform':
            ubins = np.arange(np.min(data),np.max(data)+1)
            h = np.histogram(data,bins=ubins)[0]
            csum = np.cumsum( h.astype(float) )
            yb = np.arange(0,csum[-1],step=float(csum[-1])/(self.nbins+1))
            self.qbins = np.interp(yb,csum,(ubins[:-1]+ubins[1:])/2.)

        elif self.style == 'santafe':
            ubins = np.arange(np.min(data),np.max(data)+1)
            h = np.histogram(data,bins=ubins)[0]
            csum = np.cumsum( h.astype(float) + float(knob*np.mean(h)))
            yb = np.arange(0,csum[-1],step=float(csum[-1])/float(self.nbins+1),dtype=float)
            self.qbins = np.interp(yb,csum,(ubins[:-1]+ubins[1:])/2.)

        elif self.style=='fusion': # careful, this depends on the params.expand I believe
            ubins = np.arange(np.min(data),np.max(data)+1)

        elif self.style=='bees': # careful, this depends on the params.expand I believe
            ubins = np.arange(np.min(data),np.max(data)+1)
            h = np.histogram(data,ubins)[0]
            B = (ubins[:-1]+ubins[1:])/2.
            inds = np.where(h>0)
            x = np.log2(B[inds])
            y = np.log2(h[inds])
            x0,theta = fitpoly(x,y,7)
            inds = np.where((B<(1<<10)) * (h>0))
            ym = np.mean(np.log2(B[inds]))
            distro = np.power(2.,fitcurve(np.log2(B)-x0,theta))
            if plotting:
                plt.loglog(B,h,'.')
                plt.loglog(B,distro)
                plt.grid()
                plt.title('power = %.3f'%theta[1])
                plt.show()
            csum = np.cumsum(distro)
            yb = np.arange(0,csum[-1],step=float(csum[-1])/(self.nbins+1))
            self.qbins = np.interp(yb,csum,B)

        elif self.style == 'uniform':
            mx = np.max(data)+1
            mn = np.min(data)
            self.qbins = np.arange(mn,mx,step=float(mx - mn)/float(self.nbins+1))

        elif self.style == 'wave':
            if self.wave:
                ubins = np.arange(data[0].shape[0]+1)
                wave = np.mean(data,axis=0)
                wave -= np.min(wave)
                #plt.plot(wave)
                #plt.show()
                csum = np.cumsum(wave)
                yb = np.arange(0,csum[-1],step=float(csum[-1])/(self.nbins+1))
                self.qbins = np.interp(yb,csum,np.arange(csum.shape[0]))
            else:
                print('attempting to use waveform version of quantizer on non wave style.')
        else:
            print('no style for quantizqation specified')
        return self

    def iswave():
        return bool(self.wave)

    def getbins(self,ens):
        out = []
        for e in ens:
            b = 0
            while (self.qbins[b]<e and b<self.nbins-1):
                b += 1
            out += [b]
        return out

    def getbin(self,e):
        b = 0
        while (self.qbins[b]<e and b<self.nbins-1):
            b += 1
        return b

    def getnbins(self):
        return self.nbins

    def histogram(self,data):
        if self.wave:
            j = 0
            h = np.zeros(self.qbins.shape[0]-1,dtype=float)
            for i in range(self.qbins.shape[0]-2):
                while j<self.qbins[i+1] and j<data.shape[0]:
                    h[i] += data[j]  # / float(self.qbins[i+1]-self.qbins[i]) # keep it as the integral of signal inside the bin window... convert to probability later using the bins stored in .h5
                    j+=1
            return h
        else:
            return np.histogram(data,bins=self.qbins)[0]

    def bincenters(self):
        return (self.qbins[:-1] + self.qbins[1:])/2.0
    def binedges(self):
        return self.qbins
    def binwidths(self):
        return self.qbins[1:]-self.qbins[:-1]

    def getstyle(self):
        return self.style

    @classmethod
    def aggregateBatchesH5(cls,fname,klist,qdict,agrtype='kmeans'): # based on KMeans, but probably this would be best done with using the cumulative sum as an integer implementation.
        with h5py.File(fname,'a') as f:
            for k in klist:
                bkeys = [b for b in qdict[k].keys() if not re.search('agr',b)]
                nbins = qdict[k][bkeys[0]].nbins+1
                d = np.zeros((nbins,len(bkeys)),dtype=float)
                res = np.zeros((nbins,),dtype=float)
                for i,b in enumerate(bkeys):
                    d[:,i] = qdict[k][b].qbins
                if agrtype=='kmeans':
                    qmeans = np.mean(d,axis=1)
                    kmeans = KMeans(n_clusters=nbins,init=qmeans.reshape(-1,1),n_init='auto')
                    res = kmeans.fit(d.reshape(-1,1)).cluster_centers_.reshape(-1)
                    dsname = 'agr_kmeans'
                else: # do the quick method with cumsum
                    inds = d.reshape(-1).astype(int)
                    minind = np.min(inds)
                    inds -= minind
                    h = np.zeros((np.max(inds)+1,),dtype=int)
                    for i in inds:
                        h[i] += 1
                    c = np.cumsum(h)
                    v = np.arange(len(c))
                    #y = np.zeros((nbins),dtype=float)
                    y = np.arange(len(bkeys)//2,(nbins*len(bkeys)+len(bkeys)//2),len(bkeys))
                    res = np.interp(y,c,v) + minind
                    dsname = 'agr_quick'

                ds = f[k].create_dataset(dsname,data=res)
                ds.attrs.create('nbins',qdict[k][b].nbins,dtype=np.uint32)
                ds.attrs.create('style',qdict[k][b].style)
        return

    @classmethod
    def saveBatchesH5(cls,fname,klist,qdict):
        with h5py.File(fname,'w') as f:
            for k in klist:
                grp = f.create_group(k)
                for b in qdict[k].keys():
                    ds = grp.create_dataset('qbins_%s'%b,data=qdict[k][b].qbins,dtype=float)
                    ds.attrs.create('nbins',qdict[k][b].nbins,dtype=np.uint32)
                    ds.attrs.create('style',qdict[k][b].style)
        return 

    @classmethod
    def saveH5(cls,fname,klist,qdict):
        with h5py.File(fname,'w') as f:
            for k in klist:
                grp = f.create_group(k)
                ds = grp.create_dataset('qbins',data=qdict[k].qbins,dtype=float)
                ds.attrs.create('nbins',qdict[k].nbins,dtype=np.uint32)
                ds.attrs.create('style',qdict[k].style)
        return 

    def copybins(self,bins):
        self.qbins = bins
        return self

    @classmethod
    def loadH5(cls,fname):
        with h5py.File(fname,'r') as f:
            q = cls(style=f['qbins'].attrs['style'],nbins=f['qbins'].attrs['nbins'])
            q.copybins(f['qbins'][()])
        return q

def main():
    if len(sys.argv)<2:
        print('syntax: Quantizer.py <nbins> <fname> <opt plotting? [Tt]rue>')
        return
    plotting = False
    if len(sys.argv)>2:
        if (sys.argv[-1] == 'True' or sys.argv[-1]=='true'):
            plotting = True

    fname = sys.argv[2]
    data = {} 
    quants = {}
    with h5py.File(fname,'r') as f:
        portkeys = [k for k in f.keys() if re.search('port',k)]
        for k in portkeys:
            quants[k] = Quantizer(style='nonuniform',nbins=np.uint32(sys.argv[1]))
            quants[k].setbins(f[k]['tofs'][()])
            print(len(quants[k].bincenters()))
            data[k] = quants[k].histogram(f[k]['tofs'][()])

    if plotting:
        for k in list(quants.keys())[:1]:
            #plt.plot(data[k],'.')
            #plt.plot(quants[k].bincenters(),data[k]/quants[k].binwidths(),'.')
            plt.step(quants[k].bincenters(),1000./quants[k].binwidths(),linefmt='b-',markerfmt=' ')
        plt.show()
        
    return

if __name__ == '__main__':
    main()
