from typing import List, Optional
from typing_extensions import Annotated

import h5py
import typer

from ..correlator import scatterquantize, EventPackedQuantizer, correl, xp, create_indices

run = typer.Typer(pretty_exceptions_enable=False)

ids = [  0,  22,  45, 112,
       180, 337, 135, 157,
        90, 202, 225, 247,
       270,  67, 315, 292 ]

def shared_event_count(data):
    # Print out number of events that simultaneously activated
    # each pair of detectors.
    indices = create_indices([d['events'] for d in data])
    print( len(indices) )
    for i, u in enumerate(indices):
        for j, v in enumerate(indices):
            if j < i:
                continue
            share = len(set(u) & set(v))
            print(f"{i},{j} -- {share}")

@run.command()
def quant(fname: str, detectors: Annotated[Optional[List[int]],
                                    typer.Argument()] = None,
          start: int = 4800, stop: int = 9000, nbins:int = 1000):
    P = EventPackedQuantizer(start, stop, nbins)
    f = h5py.File(fname)['mrco_hsd']

    if len(detectors) > 0: # use only the specified detectors
        data = [f[str(d)] for d in detectors]
    else: # use all detectors
        data = [f[str(i)] for i in ids]

    nquant = len(data)
    quantizers = [P]*nquant

    ans = scatterquantize(data, quantizers, join_key='events')
    N = ans.shape[1]
    C = xp.zeros((N, N))
    correl(C, ans, ans)
    print(C.shape)

    C = C.reshape((nquant,nbins, nquant,nbins))
    print(C.sum(3).sum(1))

    out="correl.npy"
    print(f"Saving results to {out}")
    xp.save(out, C)#np.transpose(C, (0,2,1,3)))
