from typing import Optional
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

# Detector layout for plots:
ids = [  0,  22,  45, 112,
       180, 337, 135, 157,
        90, 202, 225, 247,
       270,  67, 315, 292 ]

def load_h5(buf: bytes) -> Batch:
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

def tof_hist(fname: Annotated[Optional[Path], typer.Argument()] = None,
             dial: Annotated[Optional[str], typer.Option()] = None,
             start: Optional[int] = 4500,
             stop: Optional[int] = 9000,
             nbins: Optional[int] = 1000,
            ):

    if fname is not None:
        with h5py.File(fname) as f:
            batch = Batch.from_h5(f)
            src = stream.Source([batch])
    elif dial is not None:
        src = puller(dial, 1) >> stream.map(load_h5) >> stream.filter(lambda x: x is not None)
    else:
        raise ValueError("Must be run with either fname or --dial")

    """
    ids = [det[1] for det in batch.keys() if det[0] == 'mrco_hsd']
    ids.sort()
    """
    dets = [('mrco_hsd', i) for i in ids]

    x = get_xval(start, stop, nbins)
    for (H, nev) in src >> accum_tofs(dets, start, stop, nbins):
        plot_counts(x, H, nev, dets)

    #tof_counts = np.zeros(30, dtype=int)
@stream.stream
def accum_tofs(src, dets, start, stop, nbins):
    nev = {idx: 0 for idx in dets}
    H   = {idx: 0 for idx in dets}
    for i, batch in enumerate(src):
        evt = len(batch[('gmd',0)]['events'])
        
        info = [f"Batch {i}: {evt} events", ""]
        n = 0
        for idx in dets:
            N = len(batch[idx]['events'])
            tofs = batch[idx]['tofs']
            nev[idx] += N

            if N != len(tofs):
                print(f"Mismatch between events ({N}) and tofs ({len(tofs)})!")
                continue
            nev[idx] += N

            # print counts
            info[-1] += " %8d"%len(tofs)
            n += 1
            if n%4 == 0:
                info.append("")

            tofs = tofs[np.where(tofs<(1<<14))]
            
            H[idx] = H[idx]+create_hist(tofs, start, stop, nbins)

        print("\n".join(info), flush=True)
        yield H, nev

def plot_counts(x, hists, nev, ports):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(ncols=4, nrows=4, figsize=(10, 5),
                            layout="constrained")
    for i, idx in enumerate(ports):
        H = hists[idx]/nev[idx]
        total = H.sum()
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
        ax.set_xlabel(f'{idx[1]}: {total:0.1f}')
        #ax.set_ylabel('hits')
        #plt.show()

    print("Saving tof_hist.svg", flush=True)
    plt.savefig("tof_hist.svg")

def run():
    typer.run(tof_hist)
