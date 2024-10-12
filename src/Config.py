#!/sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh

import h5py
import numpy as np

class Config:
    def __init__(self):
        self.hsdchannels = {}
        self.params = {}
        self.devices = []
        self.params = {'chans':{},'t0s':{},'logicthresh':{},'offsets':{}}
        self.chans = {0:3,1:9,2:11,4:10,5:12,12:5,13:6,14:8,15:2,16:13}
        self.vlsthresh = 1000
        self.vlswin = (1024,2048)
        self.l3offset = 5100
        self.params.update({'expand':4,'inflate':2})    # expand controls the fractional resolution for scanedges by scaling index values and then zero crossing round to intermediate integers.
                                                        # inflate pads the DCT(FFT) with zeros, artificially over sampling the waveform

        self.t0s = {0:4577,1:4186,2:4323,4:4050,5:4128,12:4107,13:4111,14:4180,15:4457,16:4085} # these are not accounting for the expand nor inflate, digitizer units, 6GSps, so 6k = 1usec
        self.logicthresh = {0:-1*(1<<15), 1:-1*(1<<15), 2:-1*(1<<15), 4:-1*(1<<15), 5:-1*(1<<15), 12:-1*(1<<15), 13:-1*(1<<15), 14:-1*(1<<15), 15:-1*(1<<15), 16:-1*(1<<15)} # set by 1st knee (log-log) in val histogram
        for k in self.logicthresh.keys():
            self.logicthresh[k] >>= 2
        self.offsets = {}
        _= [self.offsets.update({k:[0]*4}) for k in self.logicthresh.keys()]


        self.hsdchannels.update({'mrco_hsd_0':'hsd_1B_A',
            'mrco_hsd_22':'hsd_1B_B',
            'mrco_hsd_45':'hsd_1A_A',
            'mrco_hsd_67':'hsd_1A_B',
            'mrco_hsd_90':'hsd_3E_A',
            'mrco_hsd_112':'hsd_3E_B',
            'mrco_hsd_135':'hsd_3D_A',
            'mrco_hsd_157':'hsd_89_B',
            'mrco_hsd_180':'hsd_01_A',
            'mrco_hsd_202':'hsd_01_B',
            'mrco_hsd_225':'hsd_DA_A',
            'mrco_hsd_247':'hsd_DA_B',
            'mrco_hsd_270':'hsd_B2_A',
            'mrco_hsd_292':'hsd_B2_B',
            'mrco_hsd_315':'hsd_B1_A',
            'mrco_hsd_337':'hsd_B1_B'} )
        return None

    def fillconfigs(self,cfgname:str):
        with h5py.File(cfgname,'r') as f:
            self.params['inflate'] = f.attrs['inflate']
            self.params['expand'] = f.attrs['expand']
            self.params['vlsthresh'] = f.attrs['vlsthresh']
            self.params['vlswin'] = f.attrs['vlswin']
            self.params['l3offset'] = f.attrs['l3offset']
            for p in f.keys():
                m = re.search('^\w+_(\d+)$',p)
                if m:
                    k = int(m.group(1))
                    self.params['chans'][k] = f[p].attrs['hsd']
                    self.params['t0s'][k] = f[p].attrs['t0']
                    self.params['logicthresh'][k] = f[p].attrs['logicthresh']
                    self.params['offsets'][k] = f[p].attrs['offsets']
        return self

    def getparams(self):
        return self.params


    def writeconfigs(self,fname:str):
        with h5py.File(fname,'w') as f:
            f.attrs.create('expand',self.params['expand']) # expand controls the fractional resolution for scanedges by scaling index values and then zero crossing round to intermediate integers.
            f.attrs.create('inflate',self.params['inflate']) # inflate pads the DCT(FFT) with zeros, artificially over sampling the waveform
            f.attrs.create('vlsthresh',data=self.vlsthresh)
            f.attrs.create('vlswin',data=self.vlswin)
            f.attrs.create('l3offset',data=self.l3offset)
            for k in self.chans.keys():
                key = 'port_%i'%int(k)
                c = f.create_group(key)
                c.attrs.create('hsd',data=self.chans[k])
                c.attrs.create('t0',data=self.t0s[k])
                c.attrs.create('logicthresh',data=self.logicthresh[k])
                c.attrs.create('offsets',data=self.offsets[k])
        return self

