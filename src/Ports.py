import numpy as np
from scipy.fftpack import dct,dst,rfft,irfft,fft,ifft
from utils import mypoly,tanhInt,tanhFloat,randomround,quick_mean,cfdLogic,cfdLogic_mod,fftLogic_fex,fftLogic,fftLogic_f16
import h5py
import time
from typing import Type,List

import matplotlib.pyplot as plt
    

class Port:
    # Note that t0s are aligned with 'prompt' in the digitizer logic signal
    # Don't forget to multiply by inflate, also, these look to jitter by up to 1 ns
    # hard coded the x4 scale-up for the sake of filling int16 dynamic range with the 12bit vls data and finer adjustment with adc offset correction

    def __init__(self,portnum,hsd,inflate=1,expand=1,nrollon=256,nrolloff=256,nadcs=4,t0=0,baselim=1<<6,logicthresh=-1*(1<<10)): # expand is for sake of Newton-Raphson
        self.rng = np.random.default_rng( time.time_ns()%(1<<8) )
        self.sampleEvery = 1000
        self.portnum = portnum
        self.hsd = hsd
        self.t0 = t0
        self.nadcs = nadcs
        self.baseshift = 1<<8
        self.baselim = baselim
        self.baseline = np.uint32(1<<8)
        self.logicthresh = logicthresh
        self.initState = True
        self.inflate = inflate
        self.expand = expand
        self.nrollon = nrollon
        self.nrolloff = nrolloff
        self.tofs = []
        self.slopes = []
        self.addresses = []
        self.nedges = []
        self.raw = {}
        self.waves = {}
        self.logics = {}
        self.shot = int(0)
        self.processAlgo = 'fex2hits' # add as method to set the Algo for either of 'fex2hits', 'fex2coeffs', or just 'wave'

        self.e:List[np.uint32] = []
        self.de:List[np.int32] = []
        self.ne = 0
        self.r = []

        self.runkey = 0
        self.name = 'hsd'

    def get_runstr(self):
        return 'run_%04i'%self.runkey

    def get_runkey(self):
        return self.runkey

    def reset(self):
        self.tofs.clear()
        self.addresses.clear()
        self.nedges.clear()
        self.slopes.clear()
        self.e:List[np.uint32].clear()
        self.de:List[np.int32].clear()
        self.ne = 0
        self.r.clear()


    @classmethod
    def slim_update_h5(cls,f,port,hsdEvents):
        print('slim_update_h5() needs to inherit only the hits and the params and only if fex/counting mode.\nCurrent mode will need to report all fex windows until CPA is run.')
        return

    @classmethod
    def update_h5(cls,f,port,hsdEvents):
        rkeys = port.keys()
        for rkey in rkeys:
            #print(rkey)
            names = port[rkey].keys()
            for name in names:
                #print(name)
                testkey = [k for k in port[rkey][name].keys()][0]
                rkeystr = port[rkey][name][testkey].get_runstr()
                rgrp = None
                nmgrp = None
                if rkeystr in f.keys():
                    rgrp = f[rkeystr]
                else:
                    rgpr = f.create_group(rkeystr)
                if name in f[rkeystr].keys():
                    nmgrp = f[rkeystr][name]
                else:
                    nmgrp = f[rkeystr].create_group(name)
        
                p = port[rkey][name]
                for key in p.keys(): # remember key == port number
                    #print(key)
                    g = None
                    if 'port_%i'%(key) in nmgrp.keys():
                        g = nmgrp['port_%03i'%(key)]
                        rawgrp = g['raw']
                        wvgrp = g['waves']
                        lggrp = g['logics']
                    else:
                        g = nmgrp.create_group('port_%03i'%(key))
                        rawgrp = g.create_group('raw')
                        wvgrp = g.create_group('waves')
                        lggrp = g.create_group('logics')
                    g.create_dataset('tofs',data=p[key].tofs,dtype=np.uint64) 
                    g.create_dataset('slopes',data=p[key].slopes,dtype=np.int64) 
                    g.create_dataset('addresses',data=p[key].addresses,dtype=np.uint64)
                    g.create_dataset('nedges',data=p[key].nedges,dtype=np.uint64)
                    for k in p[key].waves.keys():
                        rawgrp.create_dataset(k,data=p[key].raw[k].astype(np.uint16),dtype=np.uint16)
                        wvgrp.create_dataset(k,data=p[key].waves[k].astype(np.int16),dtype=np.int16)
                        lggrp.create_dataset(k,data=p[key].logics[k].astype(np.int32),dtype=np.int32)
                    g.attrs.create('inflate',data=p[key].inflate,dtype=np.uint8)
                    g.attrs.create('expand',data=p[key].expand,dtype=np.uint8)
                    g.attrs.create('t0',data=p[key].t0,dtype=float)
                    g.attrs.create('logicthresh',data=p[key].logicthresh,dtype=np.int32)
                    g.attrs.create('hsd',data=p[key].hsd,dtype=np.uint8)
                    #g.attrs.create('size',data=p[key].sz*p[key].inflate,dtype=np.uint64) ### need to also multiply by expand #### HERE HERE HERE HERE
                    g.create_dataset('events',data=hsdEvents)
        print('leaving Port.update_h5()')
        return 
        
    def set_logicthresh(self,v:np.int32):
        self.logicthresh = v
        return self

    def get_logicthresh(self):
        return self.logicthresh

    def get_runkey(self):
        return self.runkey

    def get_name(self):
        return self.name

    def set_runkey(self,r:int):
        self.runkey = r
        return self

    def set_name(self,n:str):
        self.name = n
        return self

    def addeverysample(self,o,w,l):
        eventnum = len(self.addresses)
        self.raw.update( {'shot_%i'%eventnum:np.copy(o)} )
        self.waves.update( {'shot_%i'%eventnum:np.copy(w)} )
        self.logics.update( {'shot_%i'%eventnum:np.copy(l)} )
        return self

    def addsample(self,o,w,l):
        eventnum = len(self.addresses)
        if eventnum<100:
            if eventnum%10<10: 
                self.raw.update( {'shot_%i'%eventnum:np.copy(o)} )
                self.waves.update( {'shot_%i'%eventnum:np.copy(w)} )
                self.logics.update( {'shot_%i'%eventnum:np.copy(l)} )
        elif eventnum<1000:
            if eventnum%100<10: 
                self.raw.update( {'shot_%i'%eventnum:np.copy(o)} )
                self.waves.update( {'shot_%i'%eventnum:np.copy(w)} )
                self.logics.update( {'shot_%i'%eventnum:np.copy(l)} )
        elif eventnum<10000:
            if eventnum%1000<10: 
                self.raw.update( {'shot_%i'%eventnum:np.copy(o)} )
                self.waves.update( {'shot_%i'%eventnum:np.copy(w)} )
                self.logics.update( {'shot_%i'%eventnum:np.copy(l)} )
        else:
            if eventnum%10000<10: 
                self.raw.update( {'shot_%i'%eventnum:np.copy(o)} )
                self.waves.update( {'shot_%i'%eventnum:np.copy(w)} )
                self.logics.update( {'shot_%i'%eventnum:np.copy(l)} )
        return self



    def scanedges_simple(self,d):
        tofs = []
        slopes = []
        sz = d.shape[0]
        i:int = int(10)
        while i < sz-10:
            while d[i] > self.logicthresh:
                i += 1
                if i==sz-10: return tofs,slopes,len(tofs)
            while i<sz-10 and d[i]<0:
                i += 1
            stop = i
            ''' dx / (Dy) = dx2/dy2 ; dy2*dx/Dy - dx2 ; x2-dx2 = stop - dy2*1/Dy'''
            x0 = float(stop) - float(d[stop])/float(d[stop]-d[stop-1])
            i += 1
            v = float(self.expand)*float(x0)
            tofs += [np.uint32(randomround(v,self.rng))] 
            slopes += [d[stop]-d[stop-1]] 
        return tofs,slopes,np.uint64(len(tofs))

    def scanedges(self,d):
        tofs = []
        slopes = []
        sz = d.shape[0]
        newtloops = 6
        order = 3 # this should stay fixed, since the logic zeros crossings really are cubic polys
        i = 10
        while i < sz-10:
            while d[i] > self.logicthresh:
                i += 1
                if i==sz-10: return tofs,slopes,len(tofs)
            while i<sz-10 and d[i]<d[i-1]:
                i += 1
            start = i-1
            i += 1
            while i<sz-10 and d[i]>d[i-1]:
                i += 1
            stop = i
            i += 1
            if (stop-start)<(order+1):
                continue
            x = np.arange(stop-start,dtype=float) # set x to index values
            y = d[start:stop] # set y to vector values
            x0 = float(stop)/2. # set x0 to halfway point
            #y -= (y[0]+y[-1])/2. # subtract average (this gets rid of residual DC offsets)
    
            theta = np.linalg.pinv( mypoly(np.array(x).astype(float),order=order) ).dot(np.array(y).astype(float)) # fit a polynomial (order 3) to the points
            for j in range(newtloops): # 3 rounds of Newton-Raphson
                X0 = np.array([np.power(x0,int(k)) for k in range(order+1)])
                x0 -= theta.dot(X0)/theta.dot([i*X0[(k+1)%(order+1)] for k in range(order+1)]) # this seems like maybe it should be wrong
            tofs += [float(start + x0)] 
            #X0 = np.array([np.power(x0,int(i)) for k in range(order+1)])
            #slopes += [np.int64(theta.dot([i*X0[(i+1)%(order+1)] for i in range(order+1)]))]
            slopes += [float((theta[1]+x0*theta[2])/2**18)] ## scaling to reign in the obscene derivatives... probably shoul;d be scaling d here instead
        return tofs,slopes,np.uint32(len(tofs))

    def test(self,s):
        if type(s) == type(None):
            return False
        return True

    def process(self,s,x=0):
        if self.processAlgo =='fex2coeffs':
            return process_fex2coeffs(s,x)
        elif self.processAlgo == 'fex2hits':
            return self.process_fex2hits(s,x)
        return process_wave(s,x=0)

    def process_fex2coeffs(self,s,x):
        print('HERE HERE HERE HERE')

        return True


    def advance_event(self):
        self.e = []
        self.de = []
        self.ne = 0
        self.r = []
        return self

    def set_baseline(self,val):
        self.baseline = np.uint32(val)
        return self

    def process_fex2hits(self,slist,xlist):
        baseline = 1<<13
        thise = []
        thisde = []
        r = []
        goodlist = [bool(type(s)!=type(None)) for s in slist]
        if not all(goodlist):
            print(goodlist) 
            return False
        elif len(slist)>2:
            for i,s in enumerate(slist[:-1]):
                expandBits = 1
                e,de,ne,logic = cfdLogic_mod(s,thresh=int(-1024),offset=2,expandBits=expandBits) # scan the logic vector for hits
                r = [0]*(len(logic)<<expandBits)
                for ind in e:
                    r[ind] = 1

                if True and len(self.addresses)%self.sampleEvery==0:
                    self.addsample(r,s,logic)

                start = xlist[i]<<expandBits
                thise += [start+v for v in e] 
                thisde += [d for d in de]

        if self.initState:
            self.addresses = [np.uint64(0)]
            self.nedges = [np.uint16(len(thise))]
            if len(thise)>0:
                self.tofs += thise
                self.slopes += thisde
            self.intiState = False
        else:
            self.addresses += [np.uint64(len(self.tofs))]
            self.nedges += [np.uint16(len(thise))]
            if len(thise)>0:
                self.tofs += thise
                self.slopes += thisde
        return True


    def process_wave(self,slist,xlist=[0]):
        e:List[np.int32] = []
        de = []
        ne = 0
        r = []
        s = slist[0]
        x = xlist[0]
        if type(s) == type(None):
            #self.addsample(np.zeros((2,),np.int16),np.zeros((2,),np.float16))
            e:List[np.int32] = []
            de = []
            ne = 0
            return False
        else:
            if len(self.addresses)%100==0:
                r = np.copy(s).astype(np.uint16)
            for adc in range(self.nadcs): # correcting systematic baseline differences for the four ADCs.
                b = np.mean(s[adc:self.baselim+adc:self.nadcs])
                s[adc::self.nadcs] = (s[adc::self.nadcs] ) - np.int32(b)
            #logic = fftLogic(s,inflate=self.inflate,nrolloff=self.nrolloff) #produce the "logic vector"
            logic = fftLogic_f16(s,inflate=self.inflate,nrolloff=self.nrolloff) #produce the "logic vector"
            e,de,ne = self.scanedges_simple(logic) # scan the logic vector for hits
        self.e = e
        self.de = de
        self.ne = ne

        if self.initState:
            self.addresses = [np.uint64(0)]
            self.nedges = [np.uint64(ne)]
            if ne>0:
                self.tofs += self.e
                self.slopes += self.de
        else:
            self.addresses += [np.uint64(len(self.tofs))]
            self.nedges += [np.uint64(ne)]
            if ne>0:
                self.tofs += self.e
                self.slopes += self.de
        if len(self.addresses)%100==0:
            self.addsample(r,s,logic)
        return True

    def set_initState(self,state=True):
        self.initState = state
        return self

    def print_tofs(self):
        print(self.tofs)
        print(self.slopes)
        return self

    def getnedges(self):
        if len(self.nedges)==0:
            return 0
        return self.nedges[-1]
    def setRollOn(self,n):
        self.nrollon = n
        return self
    def setRollOff(self,n):
        self.nrolloff = n
        return self

