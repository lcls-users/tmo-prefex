#!/usr/bin/python3

import h5py
import numpy 
from argon.calib.auger import transform
import argparse

parser = argparse.ArgumentParser(description = 'Applies the calibration to a given run (for now Argon Augers near 200eV only)')
parser.add_argument('-ifname',type=str,required=True,help='input filename to apply calibration')
parser.add_argument('-cfname',type=str,required=True,help='calibration file to be applied, e.g. the resulting file from src/argon.calib.augers.py')
parser.add_argument('-ret',type=str,required=True,help='retardation setting... could eventually grab this from XTC, but not doing so now')
parser.add_argument('-phEn',type=str,required=True,help='photon energy setting... could eventually grab this from XTC, but not doing so now')


def main():
    args,unk = parser.parse_known_args()
    if len(unk)>0:
        print('found unknown args')
        return
    
    histlist = []
    with h5py.File(args.ifname) as ffile:
        bins = np.linspace(150,250,800,endpoint=True)
        ret = args.ret
        phEn = args.phEn
        with h5py.File(cname,'r') as cfile:
            for p in ffile.keys():
                theta=cfile[p][args.ret][args.phen]['theta'][()]
                x = (np.log2(f[p]['tofs'] - cfile[p][ret][phEn]['theta'].attrs['toffset']) - cfile[p][ret][phEn]['theta'].attrs['xmean'])*cfile[p][ret][phEn]['theta'].attrs['scale']
                xwin = x[np.where(np.abs(x)<5)]
                X = np.stack((np.ones(xwin.shape),xwin,np.power(xwin,int(2))),axis=-1)
                theta=cfile[p][ret][phEn]['theta'][()]
                y = np.dot(X,theta)/float(cfile[p][ret][phEn]['theta'].attrs['scale']) + cfile[p][ret][phEn]['theta'].attrs['ymean']
                histlist += list(np.histogram(2**y,bins = bins)[0])
    HERE HERE HERE HERE... output the list of histograms!
    np.savetxt()
    return

if __name__ == '__main__':
    main()
