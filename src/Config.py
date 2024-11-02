#!/sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh

import h5py
import numpy as np

class Config:
    def __init__(self,is_fex=False):
        self.hsdchannels = {}
        self.params = {'devices':[]}
        self.params.update({'spectthresh':1<<5})
        self.params.update({'spectwin':(1<<7)})
        self.params.update({'l3offset':5100})
        self.params.update({'expand':2,'inflate':2})    # expand controls the fractional resolution for scanedges by scaling index values and then zero crossing round to intermediate integers.
                                                        # inflate pads the DCT(FFT) with zeros, artificially over sampling the waveform
        self.params.update({'logicthresh':{0:-1*(1<<15), 1:-1*(1<<15), 2:-1*(1<<15), 4:-1*(1<<15), 5:-1*(1<<15), 12:-1*(1<<15), 13:-1*(1<<15), 14:-1*(1<<15), 15:-1*(1<<15), 16:-1*(1<<15)}}) # set by 1st knee (log-log) in val histogram
        self.params.update({'offsets':{}})
        for k in self.params['logicthresh'].keys():
            self.params['logicthresh'][k] >>= 2
            self.params['offsets'].update({k:[0]*4})

        self.is_fex = is_fex
        self.params.update({'is_fex':self.is_fex})

        '''
        self.params.update({'hsdchannels':{'mrco_hsd_0':'hsd_1B_A',
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
            'mrco_hsd_337':'hsd_B1_B'} })
        '''
        return None

    def fillconfigs_fromH5(self,cfgname:str):
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
                    #self.params['chans'][k] = f[p].attrs['hsd']
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
            f.attrs.create('vlsthresh',data=self.params['vlsthresh'])
            f.attrs.create('vlswin',data=self.params['vlswin'])
            f.attrs.create('l3offset',data=self.params['l3offset'])
            '''
            for k in self.params['chans'].keys():
                key = 'port_%i'%int(k)
                c = f.create_group(key)
                c.attrs.create('hsd',data=self.params['chans'][k])
                c.attrs.create('t0',data=self.params['t0s'][k])
                c.attrs.create('logicthresh',data=self.params['logicthresh'][k])
                c.attrs.create('offsets',data=self.params['offsets'][k])
                '''
        return self

