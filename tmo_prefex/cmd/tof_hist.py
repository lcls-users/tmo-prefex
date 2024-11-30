from typing import Optional, List
from typing_extensions import Annotated
import io
from pathlib import Path

import numpy as np
import typer
import h5py
import stream

from lclstream.nng import puller
from lclstream.stream_utils import clock

from ..combine import Batch
from ..stream_utils import split
from .correlate import shared_event_count

run = typer.Typer(pretty_exceptions_enable=False)

# Detector layout for plots:
ids = [  0,  22,  45, 112,
       180, 337, 135, 157,
        90, 202, 225, 247,
       270,  67, 315, 292 ]

def load_h5_file(fname) -> Batch:
    with h5py.File(fname) as h:
        return Batch.from_h5(h)

def load_h5_buf(buf: bytes) -> Batch:
    try:
        with io.BytesIO(buf) as f:
            with h5py.File(f, 'r') as h:
                return Batch.from_h5(h)
    except (IOError, OSError):
        print(f"Error reading h5 from buffer with length {len(buf)}", flush=True)
        #with open("dat.h5", "wb") as f:
        #    f.write(buf)
    return None

def get_xval(start, stop, nbins):
    return np.arange(0.5, nbins)*(stop-start)/nbins + start

def create_hist(tofs, start, stop, nbins):
    #counts, _ = np.histogram(tofs, bins=nbins, range=(0,nbins))
    counts, _ = np.histogram(tofs, bins=nbins, range=(start,stop))
    return counts

@run.command()
def tof_hist(files: Annotated[Optional[List[Path]], typer.Argument()] = None,
             dial: Annotated[Optional[str], typer.Option()] = None,
             start: Optional[int] = 4500,
             stop:  Optional[int] = 9000,
             nbins: Optional[int] = 1000,
            ):

    if files is not None:
        src = stream.Source(files) >> stream.map(load_h5_file)
    elif dial is not None:
        src = puller(dial, 1) >> stream.map(load_h5_buf) \
                    >> stream.filter(lambda x: x is not None)
    else:
        raise ValueError("Must be run with either fname or --dial")

    """
    ids = [det[1] for det in batch.keys() if det[0] == 'mrco_hsd']
    ids.sort()
    """
    dets = [('mrco_hsd', i) for i in ids]

    x = get_xval(start, stop, nbins)
    for (tofs, ned, AC) in src >> split(accum_tofs(dets, start, stop, nbins),
                                    accum_edges(dets),
                                    accum_shared(dets)):
        H, nev, M = tofs
        plot_counts(x, H, M, dets)
        plot_ned(ned)
        np.save("shared.npy", AC[0])
        np.save("ncorrel.npy", AC[1])

@stream.stream
def accum_shared(src, dets):
    A = 0
    C = 0
    for batch in src:
        X, Y = shared_event_count([batch[d] for d in dets])
        A = A + X
        C = C + Y
        yield A, C

@stream.stream
def accum_edges(src, dets, nbins=30):
    """ Accumulate a histogram over values of nedges.
    """
    ned = {idx: np.zeros(nbins, dtype=int) for idx in dets}
    for i, batch in enumerate(src):
        evt = len(batch[('gmd',0)]['events'])

        for idx in dets:
            nedges = batch[idx]['nedges']
            ans = np.bincount(nedges.astype(int), minlength=nbins)
            if len(ans) > nbins:
                print(f"max nedges = {len(ans)}! neglecting {ans[nbins:].sum()} high counts")
                ans = ans[:nbins]
            ned[idx] += ans
            # all events which did not define it implicitly have nedges=0
            ned[idx][0] += evt-len(nedges)
        yield ned

@stream.stream
def accum_tofs(src, dets, start, stop, nbins):
    nev = {idx: 0 for idx in dets}
    H   = {idx: 0 for idx in dets}
    M = 0
    for i, batch in enumerate(src):
        evt = len(batch[('gmd',0)]['events'])
        M += evt
        
        info = [f"Batch {i}: {evt} events", ""]
        n = 0
        for idx in dets:
            N = len(batch[idx]['events'])
            tofs = batch[idx]['tofs']
            nev[idx] += N

            # print counts
            info[-1] += " %8d"%len(tofs)
            n += 1
            if n%4 == 0:
                info.append("")

            tofs = tofs[np.where(tofs<(1<<14))]
            
            H[idx] = H[idx]+create_hist(tofs, start, stop, nbins)

        print("\n".join(info), flush=True)
        yield H, nev, M

def plot_counts(x, hists, nev, ports):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(ncols=4, nrows=4, figsize=(10, 5),
                            layout="constrained")
    for i, idx in enumerate(ports):
        H = hists[idx]/nev
        total = H.sum()*100
        #h = H.cumsum() # convert to CDF for plotting
        h = H/(x[1]-x[0]) # take derivative for PDF plotting

        ax = axs[i//4, i%4]
        #ax.title(f'{idx}')#: {ev} events, {total/ev} counts/event')
        #ax.plot(x, h, label=str(idx))
        ax.step(x, h, label=str(idx))
        #ax.legend()
        #ax.set_xlim(2800,3400)
        #ax.set_xlim(1500, 3000)
        #ax.set_xlabel('histbins = 1 hsd bins')
        ax.set_xlabel(f'{idx[1]}: {total:0.1f}/100')
        #ax.set_ylabel('hits')
        #plt.show()

    print("Saving tof_hist.svg", flush=True)
    plt.savefig("tof_hist.svg")

def plot_ned(ned):
    import matplotlib.pyplot as plt

    plt.cla()
    plt.clf()
    add = 0
    for idx, v in ned.items():
        x = np.arange(len(v))
        v[0] = 0 # zero this bin so it doesn't get in the way
        plt.step(x, v+add, label=str(idx))
        add += v.max()
    plt.legend()

    print("Saving edges_hist.svg", flush=True)
    plt.savefig("edges_hist.svg")
