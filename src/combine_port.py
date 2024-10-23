from typing import List, Dict, Any, Union

from new_port import WaveData, FexData, PortConfig

import numpy as np
from stream import filter

def load_data(cfg, data, datasets) -> Dict[str,np.ndarray]:
    ret = {'cfg': cfg}
    for name, dtype in datasets:
        ret[name] = data[name][:]
    return ret

def concat(arg):
    if isinstance(arg, list):
        vals = arg
    else:
        vals = list(arg)
    if len(vals) == 0:
        return np.array([], dtype=np.int32)
    return np.hstack(vals)

class Batch(dict):
    """ Class to hold and concatenate a dict-of-dict-of (h5-like dicts)
        as produced by save_batch.
    """
    datasets=[ ('events', np.uint32),
               ('addresses', np.uint64),
               ('nedges', np.uint64),
               ('tofs', np.uint64),
               ('slopes', np.int64),
               ('rl_events', np.uint32),
               ('rl_addresses', np.uint64),
               ('raw_lens', np.uint16),
               ('logic_lens', np.uint16),
               ('rl_data', np.int32),
             ]
    def __init__(self, data):
        super().__init__(data)

    def extend(self, *data):
        print("FIXME: need to update *addresses")
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
        #ports = {}
        data = {}
        for k1, v1 in f.items():
            #ports[k1] = {}
            data[k1] = {}
            for k2, v2 in v1.items():
                cfg = PortConfig.model_validate_json(
                                            v2.attrs["PortConfig"])
                data[k1][k2] = load_data(cfg, v2, cls.datasets)

        return cls(data)

    def write_h5(self, f):
        for hsdname, p in self.items():
            if hsdname in f.keys():
                nmgrp = f[hsdname]
            else:
                nmgrp = f.create_group(hsdname)

            for key, data in p.items(): # remember key == port number
                if 'port_%i'%(key) in nmgrp.keys():
                    g = nmgrp['port_%i'%(key)]
                else:
                    g = nmgrp.create_group('port_%i'%(key))

                for name, dtype in self.datasets:
                    g.create_dataset(name,
                                     data=np.array(data[name], dtype=dtype),
                                     dtype=dtype)

                # Store PortConfig in a json string.
                g.attrs.create("PortConfig",
                               data=data['cfg'].model_dump_json())
                #port = data['cfg']
                #g.attrs.create('size',data=port.sz*port.inflate,dtype=np.uint64) ### need to also multiply by expand #### HERE HERE HERE HERE

def should_save_raw(eventnum):
    # first 10 of every 10, then first 10 of every 100, ...
    mod = 10
    cap = 100
    while eventnum > cap:
        mod *= 10
        cap *= 10
        if cap == 100000:
            break
    return (eventnum % mod) < 10

def map_dd(u: List[Dict[str,Dict[int,Any]]],
           fn
          ) -> Dict[str,Dict[int,Any]]:
    """ Join together all values from the inputs
        under their respective dictionary key.
        
        Then run the function on each list
        to produce a dict holding the function result
        as each value.
    """
    if len(u) == 0:
        return {}

    n = len(u)
    val = { k:{k2:[None]*n for k2,v2 in v.items()}
                for k,v in u[0].items()
          }

    for i,x in enumerate(u):
        for k, v in x.items():
            for k2, v2 in v.items():
                val[k][k2][i] = v2

    ans = {}
    for k, v in val.items():
        ans[k] = {}
        for k2, v2 in v.items():
            ans[k][k2] = fn(v2)
    return ans

def save_batch(waves: Union[List[WaveData], List[FexData]]
              ) -> Dict[str,Dict[int,Any]]:
    """ Save a batch of data.
        Usually called every 100-th event or so.
    """
    events = [x.event for x in waves]
    nedges = [x.nedges for x in waves]

    # combine [raw,logic] together
    # at ea. rl_event[i], rl_addresses[i]
    # and identify raw_len[i] and logic_len[i] separately
    rl_idx = []
    rl_events = []
    rl_addresses = []
    raw_lens = []
    logic_lens = []
    k = 0
    for i, ev in enumerate(events):
        if should_save_raw(ev):
            u = len(waves[i].raw)
            v = len(waves[i].logic)
            if u+v == 0:
                continue
            rl_idx.append(i)
            rl_events.append(ev)
            rl_addresses.append(k)
            raw_lens.append(u)
            logic_lens.append(v)
            k += u+v

    return dict(
        cfg = waves[0].cfg,
        events = events,
        addresses = np.cumsum(nedges)-nedges[0],
        tofs = concat(x.tofs for x in waves),
        slopes = concat(x.slopes for x in waves),
        nedges = nedges,

        rl_events = rl_events,
        rl_addresses = rl_addresses,
        raw_lens = raw_lens,
        logic_lens = logic_lens,
        rl_data = concat(concat([waves[i].raw,waves[i].logic])
                              for i in rl_idx ),
        #waves = waves[i].raw,
    )

# Convert a batch (aka list) of dict-of-dicts to a dict-of-dict-of-(h5-like dicts)
save_dd_batch = lambda u: Batch( map_dd(u, save_batch) )
