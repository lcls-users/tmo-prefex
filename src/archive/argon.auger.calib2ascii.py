#!/usr/bin/python3

import h5py
import numpy as np
import sys

def main():
    if len(sys.argv)<3:
        print('Give me the input .h5 file and then the output .dat file')
        return
    with h5py.File(sys.argv[1],'r') as f:
        for p in f.keys():
            energies = []
            tofs = []
            vrets = []
            phens = []
            for k in f[p].keys():
                energies += list(f[p][k]['energies'][()])
                d = list(f[p][k]['tofs'][()])
                tofs += d
                vrets += [int(f[p][k]['tofs'].attrs['ret'])]*len(d)
                phens += [int(f[p][k]['tofs'].attrs['phEn'])]*len(d)
            np.savetxt('%s.%s'%(sys.argv[2],p),np.column_stack((tofs,energies,vrets,phens)),fmt='%i',header = 'tofs_bins(expand=16)\tLiteratureEnergies_eV\tretardation_eV\tphotonEnergies_eV')
    return


if __name__ == '__main__':
    main()
