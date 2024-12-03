#!/usr/bin/python3

import matplotlib.pyplot as plt
import math
import h5py
import sys
import numpy as np

def gauss(x,xc,w,a):
    return float(a)*math.exp(-1.0*((x-xc)/w)**2)

def lawrence(x,xc,w,a):
    return float(a)/(((x-xc)/w)**2 + 1)

def main():
    k = str(sys.argv[1])
    fname = str(sys.argv[2])

    data = {}
    with h5py.File(fname,'r') as f:
        _ = [data.update({k:f[k][()]}) for k in f.keys()]

    ports = [p for p in data.keys()]


    peakenergies = [207.11, 205.51, 205.09, 203.35, 203.09, 200.96]
    peakenergieslbls = ['%.2f'%v for v in peakenergies]
    #peakpositions = [151, 355, 418, 652, 684, 1024]

    peakpositions = [341, 543, 595, 837, 872, 1203]
    plt.plot(peakpositions,peakenergies,'.')
    C = [1]*len(peakpositions)
    P = [v for v in peakpositions]
    PP = [v**2 for v in peakpositions]
    X = np.stack((C,P,PP),axis=1)
    Y = np.expand_dims(np.array(peakenergies),axis=0)
    Omega = np.dot(Y,np.linalg.pinv(X).T)
    xvals0 = [1 for v in range(0,len(data[k]),200)]
    xvals1 = [v for v in range(0,len(data[k]),200)]
    xvals2 = [v**2 for v in range(0,len(data[k]),200)]

    print(Omega)
    xx = np.stack((xvals0,xvals1,xvals2),axis=1)
    yvals = [v for v in np.dot(Omega,xx.T).T]
    
    plt.plot(xvals1,np.array(yvals),'-')
    plt.show()

    eVperbin = (1.311/200)



    delta=35
    y0=20
    a=420
    w=13
    xc=837
    fig,ax1=plt.subplots(figsize=(10,5))
    ax1.plot(data[k],'.');
    ax1.plot([i+xc for i in range(-100,100)],[y0+lawrence(float(i+xc),xc,w,a)+lawrence(float(i+xc),xc+delta,w,a/5) for i in range(-100,100)],'-');
    ax1.legend(['port_045','Lorentzian,\nw=%i meV'%(int(1000*eVperbin*w*2))])
    ax1.set_xlabel('Time-of-Flight [arb. units]')
    ax1.set_ylabel('histogram [counts]')

    ax2 = ax1.twiny()
    ax2.set_xlabel('Kinetic energy [eV]')
    ax1.set_xlim(-10,data[k].shape[0]+10)
    ax2.set_xlim(-10,data[k].shape[0]+10)
    ax2.set_xticks(xvals1,['%.1f'%v for v in yvals])

    plt.savefig('forKevinPrince.png')
    plt.show()
    return

if __name__ == '__main__':
    if len(sys.argv)<3:
        print('python3 ./ArgonAugerFigure.py <port_###> <input file> ')
    else:
        main()
