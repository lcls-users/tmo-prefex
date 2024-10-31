#!/usr/bin/python3

from typing_extensions import Annotated
from pathlib import Path

import h5py
import numpy as np
import typer


def main(fname: Path):
    detstr = 'mrco_hsd'

    hists = {}
    events = {}
    nbin = 1<<13
    with h5py.File(fname,'r') as f:
        ports = f[detstr].keys()
        portnums = list(map(int, ports))
        portnums.sort()
        for i, p in enumerate(portnums):
            t = f[detstr][str(p)]['tofs'][()]
            ev = len(f[detstr][str(p)]['events'])
            tvalid = [int(v)>>1 for v in t if v<(1<<14)]
            #print(p, tvalid)
            #continue
            
            counts, _ = np.histogram(tvalid, bins=nbin, range=(0,nbin))
            if p in hists:
                hists[p] += counts
                events[p] += ev
            else:
                hists[p] = counts
                events[p] = ev

def plot_counts(hists, events):
    import matplotlib.pyplot as plt

    for p, h in hists.items():
        ev = events[p]
        h = h.cumsum() # convert to CDF for plotting
        #total = h[-1]
        #h /= total
        h /= ev # normalize to per-event counts.

        plt.title(f'port {p}: {ev} events, {total/ev} counts/event')
        plt.plot(h, label='port %i'%p)
        plt.legend()
        plt.xlim(2800,3500)
        plt.xlabel('histbins = 2 hsd bins')
        plt.ylabel('hits')
        plt.show()

def run():
    typer.run(main)

if __name__ == "__main__":
    run()
