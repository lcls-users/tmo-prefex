import numpy as np

class Gmd:
    def __init__(self):
        self.en = []
        self.runkey = int(0)
        self.initState:bool = True
        self.unit:str = 'uJ'
        self.scale:np.float16 = 1000
        self.name:str = 'gmd'
        return 

    def reset(self):
        self.en.clear()

    @classmethod
    def slim_update_h5(cls,f,gmd,gmdEvents):
        grpgmd = None
        for rkey in gmd.keys():
            if 'gmd' in f.keys():
                grpgmd = f['gmd']
            else:
                grpgmd = f.create_group('gmd')
            grpgms.attrs.create('run',data=gmd[rkey][gmdname].get_runstr(),dtype=str)
            grpgmd.create_dataset('gmdenergy',data=gmd.en,dtype=np.uint16)
            grpgmd.create_dataset('events',data=gmdEvents)
        return

    @classmethod
    def update_h5(cls,f,gmd,gmdEvents):
        grpgmd = None
        grprun = None
        for rkey in gmd.keys():
            for gmdname in gmd[rkey].keys():
                thisgmd = gmd[rkey][gmdname]
                if gmdname not in f.keys():
                    f.create_group(gmdname)
                grpgmd = f[gmdname]
                grpgmd.attrs.create('run',data=gmd[rkey][gmdname].get_runstr())
                endata = grpgmd.create_dataset('energy',data=thisgmd.en,dtype=np.uint16)
                endata.attrs.create('unit',data=thisgmd.unit)
                endata.attrs.create('scale',data=thisgmd.scale)
                grpgmd.create_dataset('events',data=gmdEvents)
        return

    def test(self,e):
        if type(e)==type(None):
            return False
        if e<0:
            return False
        return True

    def process(self,e):
        if self.initState:
            self.en = [np.uint16(e*self.scale)]
        else:
            self.en += [np.uint16(e*self.scale)]
        return True

    def get_runstr(self):
        return 'run_%04i'%self.runkey

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

    def set_initState(self,state):
        self.initState = state
        return self

    def set_unit(self,unitname='uJ',scale=1000):
        self.unit = unitname
        self.scale = scale
        return self
