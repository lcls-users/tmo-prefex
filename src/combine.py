from typing import List, Dict, Any
from collections.abc import Callable

from Hsd import PortConfig, WaveData, FexData
from Gmd import GmdConfig, GmdData

import numpy as np
from stream import filter

def load_data(ret, data) -> Dict[str,np.ndarray]:
    for name in data.keys():
        ret[name] = data[name][:]
    return ret

class Batch(dict):
    """ Class to hold and concatenate a dict-of-dict-of (h5-like dicts)
        as produced by save_batch.
    """
    dtypes = {'events': np.uint32,
              # from HSD detectors
              'PortConfig': 'json',
              'addresses': np.uint64,
              'nedges': np.uint64,
              'tofs': np.uint64,
              'slopes': np.int64,
              'rl_events': np.uint32,
              'rl_addresses': np.uint64,
              'raw_lens': np.uint16,
              'logic_lens': np.uint16,
              'rl_data': np.int32, # rwl were u16,s16,s32
              # from GMD detectors
              'GmdConfig': 'json',
              'energies': np.int16,
             }
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def extend(self, *data):
        raise NotImplementedError("keys, datasets, tbdh")
        for k1, v1 in self.items():
            for k2, v2 in v1.items():
                ret = {}
                for name, dtype in self.datasets:
                    u = [v2[name]] + [d[k1][k2][name] for d in data]
                    if name == "addresses":
                        print("FIXME")
                        exit(1)
                        off = np.cumsum([0] + [v2[name][-1]] + [d[k1][k2]["nedges"][-1] for d in data[:-1]])
                        for x, o in zip(u, off):
                            x += o
                    if name == "rl_addresses":
                        off = np.cumsum([0] + [v2[name][-1]] + [d[k1][k2]["raw_lens"][-1]+d[k1][k2]["logic_lens"][-1] for d in data[:-1]])
                        for x, o in zip(u, off):
                            x += o
                    ret[name] = concat(u)
                v1[k2] = ret

    @classmethod
    def from_h5(cls, f):
        data = cls()

        for k, v in f.items():
            # TODO: parse k as (name, int)
            #
            # TODO: determine type of detector (from k?)
            #       and gather config. options here.
            ret = {"PortConfig": PortConfig.model_validate_json(
                                        v.attrs["PortConfig"])
                  }
            data[k] = load_data(ret, v)

        return cls(data)

    def write_h5(self, f):
        for (name,chan), data in self.items():
            if name in f.keys():
                nmgrp = f[name]
            else:
                nmgrp = f.create_group(name)
            c = str(chan) # hdf5 won't accept numbers as keys
            if c in nmgrp.keys():
                g = nmgrp[c]
            else:
                g = nmgrp.create_group(c)

            for k, v in data.items():
                dtype = self.dtypes[k]
                if dtype == 'json':
                    # Store PortConfig, etc. in json strings.
                    g.attrs.create(k, data=v.model_dump_json())
                else:

                    g.create_dataset(k,
                                     data=np.array(v, dtype=dtype),
                                     dtype=dtype)

def batch_data(u: List[ List[Dict] ],
               fns: List[ Callable[ [List[Any]], Dict[str,Any]] ]
              ) -> Batch:
    """ Converts a list of (hsd, gmd, ...) data tuples to a Batch.
        fns should correspond to the type of tuples passed.

        So (hsd, gmd) should use bach_data(elems, [save_hsd,save_gmd])

        Basically transposes the indices so that the outer list
        dimension (events) is the inner one.
        
        Then, for each detector, run that detector's
        data batching function on each list.
        The result is a dict holding the function result
        as each value.
    """
    if len(u) == 0:
        return Batch()
    assert len(u[0]) == len(fns), "I must have one combination function for each detector type."

    n = len(u)    # events
    def mk_empty():
        return [None]*n

    val = {}
    for det_outputs in u[0]: # First output for ea. detector type
        val.update({ k:mk_empty() for k in det_outputs.keys() })

    for i,x in enumerate(u): # list elems (inner)
        for _, y in enumerate(x): # outputs for ea. detector type
            for k, v in y.items():
                val[k][i] = v

    ans = {}
    for fn, det_outputs in zip(fns,u[0]):
        for k in det_outputs.keys():
            ans[k] = fn( val[k] )
    return Batch(ans)
