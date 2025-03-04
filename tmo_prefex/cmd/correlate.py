from typing import List, Optional
from typing_extensions import Annotated
from pathlib import Path

import io
import stream
import h5py
import typer
import numpy as np

from lclstream.nng import puller
from ..correlator import (
    scatterquantize,
    EventPackedQuantizer,
    PassThroughQ,
    correl, xp,
)

run = typer.Typer(pretty_exceptions_enable=False)

ids = [  0,  22,  45, 112,
       180, 337, 135, 157,
        90, 202, 225, 247,
       270,  67, 315, 292 ]

def shared_event_count(data, join_key='events', vals='nedges'):
    # Tabulate number of events that simultaneously activated
    # each pair of detectors (A) as well as correlation between
    # counts in detectors, \sum_ev n_i*n_j

    N = len(data)
    A = np.zeros((N,N))
    C = np.zeros((N,N))
    for i, a in enumerate(data):
        for j, b in enumerate(data):
            if j < i:
                continue
            ea = set(a[join_key])
            eb = set(b[join_key])
            share = list(ea & eb)
            share.sort()
            nshare = len(share)

            inds = {e:i for i,e in enumerate(share)}

            x = np.zeros(nshare, dtype=a[vals].dtype)
            y = np.zeros(nshare, dtype=b[vals].dtype)
            for e,n in zip(a[join_key], a[vals]):
                if e in inds:
                    x[inds[e]] += n
            for e,n in zip(b[join_key], b[vals]):
                if e in inds:
                    y[inds[e]] += n
            v = np.dot(x, y)

            A[i,j] += nshare
            C[i,j] += v
            if i != j:
                A[j,i] += nshare
                C[j,i] += v
    return A, C

def load_h5(buf: bytes) -> h5py.File:
    try:
        f = io.BytesIO(buf)
        return h5py.File(f, 'r')
    except (IOError, OSError):
        print(f"Error reading h5 from buffer with length {len(buf)}", flush=True)

def gather(f, detectors):
    return [f[det[0]][str(det[1])] for det in detectors]

#detectors : Annotated[Optional[List[int]],
#                          typer.Argument()] = None,
@run.command()
def quant(fname: Annotated[Optional[Path], typer.Argument()] = None,
          dial: Annotated[Optional[str], typer.Option()] = None,
          start: int = 4800, stop: int = 9000, nbins:int = 1000):

    P = EventPackedQuantizer(start, stop, nbins)
    S = PassThroughQ(1024, 1024+nbins)

    detectors = [('mrco_hsd', i) for i in ids]
    nquant = len(detectors)
    quantizers = [P]*nquant
    if True:
        detectors += [('tmo_fzppiranha', 0)]
        quantizers.append(S)
        nquant += 1

    if fname is not None:
        f = h5py.File(fname)
        src = stream.Source([f])
    elif dial is not None:
        src = puller(dial, 1) >> stream.map(load_h5) \
                >> stream.filter(lambda x: x is not None)
    else:
        assert False, "Either a filename or --dial is required."
    src >>= stream.map(lambda f: gather(f, detectors))

    C = None
    for i,data in enumerate(src):
        print(f"Batch {i}", flush=True)
        ans = scatterquantize(data, quantizers, join_key='events')
        N = ans.shape[1]
        if C is None:
            C = xp.zeros((N, N))
        correl(C, ans, ans)

        cross = C.reshape((nquant,nbins, nquant,nbins)) \
                .sum(3).sum(1)
        print(cross, flush=True)

        out="correl.npy"
        print(f"Saving results to {out}")
        xp.save(out, C.reshape((nquant,nbins, nquant,nbins)))
