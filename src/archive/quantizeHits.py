#!/cds/sw/ds/ana/conda2/inst/envs/ps-4.2.5/bin/python3
import numpy as np
import sys
import h5py
import re
import matplotlib.pyplot as plt
from Quantizers import Quantizer

def main():
    if len(sys.argv)<2:
        print('syntax: quantizeHits.py <nbins> <fname> <opt plotting? [Tt]rue>')
        return
    plotting = False
    if len(sys.argv)>2:
        if (sys.argv[-1] == 'True' or sys.argv[-1]=='true'):
            plotting = True

    nsum = 2000
    fname = sys.argv[2]
    data = {} 
    quants = {}
    portkeys = []
    with h5py.File(fname,'r') as f:
        portkeys = [k for k in f.keys() if (re.search('port',k) and not re.search('_16',k) and not re.search('_2',k))]
        for k in portkeys:
            quants[k] = Quantizer(style='nonuniform',nbins=np.uint32(sys.argv[1]))
            quants[k].setbins(f[k]['tofs'][()])
            data[k] = []
            for i in range(len(f[k]['addresses'][()])//nsum - 1):
                a = f[k]['addresses'][()][nsum*i]
                #n = f[k]['nedges'][()][nsum*i]
                n = np.sum(f[k]['nedges'][()][nsum*i:nsum*(i+1)])
                data[k] += [quants[k].histogram(f[k]['tofs'][()][a:a+n])]

    m = re.search('(.*)\.h5',fname)
    if m:
        qfname = '%s.quantizers.h5'%m.group(1)
        print(qfname)
        Quantizer.saveH5(qfname,portkeys,quants)

    if plotting:
        image=np.zeros((len(data[portkeys[0]][0]),len(data.keys())),dtype=np.uint16)
        print(image.shape)
        XX,YY = np.meshgrid(np.arange(image.shape[1]),np.arange(image.shape[0]))
        print(XX.shape)
        print(YY.shape)
        _= [print(len(data[k][0])) for k in data.keys()]
        for i in range(len(data[portkeys[0]])):
            for j,k in enumerate(data.keys()):
                image[:,j] = data[k][i]
            #imname = '/reg/data/ana16/tmo/tmox42619/scratch/ryan_output_multicolorhack/ascii/sum2000_img_%i.dat'%i
            #np.savetxt(imname,image,fmt='%i')
            #plt.imshow(image)
            plt.pcolor(XX,YY,image)
            plt.show()
            #print('saving %s'%imname)
        #for k in list(data.keys())[:1]:
            #plt.plot(data[k],'.')
            #plt.plot(quants[k].bincenters(),data[k]/quants[k].binwidths(),'.')
            #plt.stem(quants[k].bincenters(),1000./quants[k].binwidths(),linefmt='b-',markerfmt=' ')
        #plt.imshow()
        
    return

if __name__=='__main__':
    main()

