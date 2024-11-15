import numpy as np

class Scan:
    def __init__(self,name='lxt'):
        self.val = []
        self.runkey = int(0)
        self.initState:bool = True
        self.scale:np.float16 = 1000
        self.name:str = name
        return 

    def reset(self):
        self.val.clear()

    @classmethod
    def slim_update_h5(cls,f,scan,scanEvents):
        grpscan = None
        if self.name in f.keys():
            grpscan = f[self.name]
        else:
            grpscan = f.create_group(self.name)
        valdata = grpscan.create_dataset('values',data=scan.val,dtype=np.int32)
        valdata.attrs.create('unit',data=thisscan.unit)
        valdata.attrs.create('scale',data=thisscan.scale)
        grpscan.create_dataset('events',data=scanEvents)
        return

    @classmethod
    def update_h5(cls,f,scan,scanEvents):
        print('updating scan for h5')
        grpscan = None
        grprun = None
        for rkey in scan.keys():
            for name in scan[rkey].keys():
                thisscan = scan[rkey][name]
                rstr = scan[rkey][name].get_runstr()
                if rstr not in f.keys():
                    f.create_group(rstr)
                if name not in f[rstr].keys():
                    f[rstr].create_group(name)
                grpscan = f[rstr][name]
                valdata = grpscan.create_dataset('values',data=thisscan.val,dtype=np.int32)
                valdata.attrs.create('unit',data=thisscan.unit)
                valdata.attrs.create('scale',data=thisscan.scale)
                grpscan.create_dataset('events',data=scanEvents,dtype=np.uint64)
        return

    def test(self,v):
        if type(v)==type(None):
            print('Damn you scan variable!')
            return False
        return True

    def process(self,v):
        if self.initState:
            self.val = [np.int16(v*self.scale)]
        else:
            self.val += [np.int16(v*self.scale)]
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

    def set_unit(self,unitname='fs',scale=1e15):
        self.unit = unitname
        self.scale = scale
        return self
