from typing import List, Dict, Any, Union

from new_port import WaveData, FexData

import numpy as np
from stream import filter

def write_h5(f, ports, port_data, hsdEvents):
    for hsdname, p in ports.items():
        if hsdname in f.keys():
            nmgrp = f[hsdname]
        else:
            nmgrp = f.create_group(hsdname)

        for key, port in p.items(): # remember key == port number
            data = port_data[hsdname][key]

            if 'port_%i'%(key) in nmgrp.keys():
                g = nmgrp['port_%i'%(key)]
            else:
                g = nmgrp.create_group('port_%i'%(key))

            for name, dtype in [
                        ('events',np.uint32),
                        ('addresses', np.uint64),
                        ('nedges', np.uint64),
                        ('tofs', np.uint64),
                        ('slopes', np.int64),
                        ('raw_events', np.uint32),
                        ('raw_addresses', np.uint64),
                        ('raw_lens', np.uint64),
                        ('raw', np.uint16),
                        #('waves', p.int16),
                        ('logics',np.int32),
                    ]:
                g.create_dataset(name,
                                 data=data[name].astype(dtype),
                                 dtype=dtype)

            g.attrs.create('inflate',data=port.inflate,dtype=np.uint8)
            g.attrs.create('expand',data=port.expand,dtype=np.uint8)
            g.attrs.create('t0',data=port.t0,dtype=float)
            g.attrs.create('logicthresh',data=port.logicthresh,dtype=np.int32)
            g.attrs.create('hsd',data=port.hsd,dtype=np.uint8)
            #g.attrs.create('size',data=port.sz*port.inflate,dtype=np.uint64) ### need to also multiply by expand #### HERE HERE HERE HERE

def should_save_raw(eventnum):
    cap = 100
    while eventnum > cap:
        cap *= 10
        if cap == 100000:
            break
    z = cap//10
    return eventnum % z < 10

def save_batch(waves: Union[List[WaveData], List[FexData]]
              ) -> Dict[str,Any]:
    """ save a batch of data
        Usually called every 100-th event or so.
    """
    events = [x.event for x in waves]
    nedges = [x.nedges for x in waves]

    raw_idx = []
    raw_events = []
    raw_addresses = []
    raw_lens = []
    k = 0
    for i, ev in enumerate(events):
        if should_save_raw(ev):
            u = len(waves[i].raw)
            raw_idx.append(i)
            raw_events.append(ev)
            raw_addresses.append(k)
            raw_lens.append(u)
            k += u

    return dict(
        events = events,
        addresses = np.cumsum(nedges)-nedges[0],
        tofs = np.hstack([x.tofs for x in waves]),
        slopes = np.hstack([x.slopes for x in waves]),
        nedges = nedges,

        raw_events = raw_events,
        raw_addresses = raw_addresses,
        raw_lens = raw_lens,
        raw = np.hstack([waves[i].raw for i in raw_idx]),
        #waves = waves[i].raw,
        logics = np.hstack([waves[i].logic for i in raw_idx]),
    )
