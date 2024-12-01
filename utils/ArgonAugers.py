#!/usr/bin/python3

import h5py
import matplotlib.pyplot as plt
import sys
import numpy as np
import re

def main():
    data = np.zeros((1100,16),dtype=np.uint16)
    with h5py.File(sys.argv[1],'r') as f:
        ports = [p for p in f.keys()]
        ports = np.sort(ports)
        for i,p in enumerate(ports):
            trace = f[p][()]
            imax = np.argmax(trace)
            vmax = trace[imax]
            data[:,i]=[(1500*v)//vmax for v in trace[imax-650:imax-650+data.shape[0]] ]
    peakenergies = ['%.2f'%v for v in [207.11, 205.51, 205.09, 203.35, 203.09, 200.96]]
    peakpositions = [151, 355, 418, 652, 684, 1024]

    fig,ax1 = plt.subplots(figsize=(8,10))
    for i in range(data.shape[1]):
        ax1.plot(data[:,i]+(1<<10)*i,'k')
    ax2 = ax1.twiny()
    ax1.set_yticks([i*(1<<11) for i in range(data.shape[1]>>1)],['%i'%(i*45) for i in range(data.shape[1]>>1)])
    ax1.set_xlabel('Time-of-Flight bin [arb]')
    ax2.set_xlabel('Kinetic energy [eV]')
    ax1.set_ylabel('Angle [deg]')
    ax1.set_xlim(-10,data.shape[0]+10)
    ax2.set_xlim(-10,data.shape[0]+10)
    ax2.set_xticks(peakpositions,peakenergies,rotation=30)
    runstr = 'r0000'
    m = re.search('(r\d+)/',sys.argv[1])
    if m:
        runstr = m.group(1)
    print('runstr = %s'%(runstr))
    plt.savefig('./figures/%s.ArAugers.png'%(runstr))
    plt.show()
    return

if __name__ == '__main__':
    if len(sys.argv)<2:
        print('ArgonAugers <h5 filename>')
    else:
        main()
