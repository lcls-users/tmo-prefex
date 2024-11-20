import numpy as np

class Evr:
    self.LASER:int = 280 # on and timed
    self.GOOSE:int = 281 # on but mistimed
    self.ANYLASER:int = 282 # on, either mistimed or timed

    def __init__(self):
        self.codes = []
        self.runkey = int(0)
        self.initState:bool = True
        return 

    def reset(self):
        self.codes.clear()

    @classmethod
    def slim_update_h5(cls,f,evr,evrEvents):
        grpevr = None
        if 'evr' in f.keys():
            grpevr = f['evr']
        else:
            grpevr = f.create_group('evr')
        data = grpevr.create_dataset('evrdata',data=evr.codes,dtype=np.uint16)
        data.attrs.create('codenames',data=['laser','goose','anylaser']
        grpevr.create_dataset('events',data=evrEvents)
        return

    @classmethod
    def update_h5(cls,f,evr,evrEvents):
        grpevr = None
        grprun = None
        for rkey in evr.keys():
            thisevr = evr[rkey]
            rstr = evr[rkey].get_runstr()
            if rstr not in f.keys():
                f.create_group(rstr)
            if 'evr' not in f[rstr].keys():
                f[rstr].create_group('evr')
            grpevr = f[rstr]['evr']
            evrdata = grpgmd.create_dataset('evrdata',data=evr.codes,dtype=bool)
            evrdata.attrs.create('stride',3,dtype=np.uint8)
            evrdata.attrs.create('codenames',data=['laser','goose','anylaser'])
            grpevr.create_dataset('events',data=evrEvents)
        return

    def test(self,e):
        if type(e)==type(None):
            return False
        if len(e)<(self.ANYLASER + 1):
            return False
        return True

    def process(self,e):
        if self.initState:
            self.codes = [bool(v) for v in e[self.LASER:self.ANYLASER+1]]
        else:
            self.codes += [bool(v) for v in e[self.LASER:self.ANYLASER+1]]
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

