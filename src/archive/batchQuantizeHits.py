#!/cds/sw/ds/ana/conda2/inst/envs/ps-4.2.5/bin/python3
import numpy as np
import sys
import h5py
import re
import time
import matplotlib.pyplot as plt
from Quantizers import Quantizer
from math import log2

def main():
    rng = np.random.default_rng()
    quantizeonly = True
    plotting = False

    if len(sys.argv)<4:
        print('syntax: batchQuantizeHits.py <nbatches> <nbins> <fname> <opt plotting? [Tt]rue>')
        return
    if len(sys.argv)>4:
        if (sys.argv[-1] == 'True' or sys.argv[-1]=='true'):
            plotting = True

    nsum = 1<<11
    nbatches = int(sys.argv[1])
    fname = sys.argv[3]
    data = {} 
    quants = {}
    portkeys = []
    with h5py.File(fname,'r') as f:
        portkeys = [k for k in f.keys() if (re.search('port',k) and not re.search('_16',k) and not re.search('_2',k))]
        for k in portkeys:
            quants.update({k:{}})
            data.update({k:{}})
            batchlist = ['b%02i'%b for b in range(nbatches)]
            tofs = np.copy(f[k]['tofs'][()])
            rng.shuffle(tofs)
            shiftntofs = int(log2(len(tofs))-log2(nbatches))
            for b,bkey in enumerate(batchlist):
                quants[k].update({bkey:Quantizer(style='nonuniform',nbins=np.uint32(sys.argv[2]))})
                quants[k][bkey].setbins(tofs[b*(1<<shiftntofs):(b+1)*(1<<shiftntofs)])
                data[k].update({bkey:[]})
                if not quantizeonly:
                    for i in range(len(f[k]['addresses'][()])//nsum - 1):
                        a = f[k]['addresses'][()][nsum*i]
                        #n = f[k]['nedges'][()][nsum*i]
                        n = np.sum(f[k]['nedges'][()][nsum*i:nsum*(i+1)])
                        data[k][bkey] += [quants[k][bkey].histogram(f[k]['tofs'][()][a:a+n])]

    m = re.search('(.*)\.h5',fname)
    if m:
        qfname = '%s.batchquantizers.h5'%m.group(1)
        print(qfname)
        Quantizer.saveBatchesH5(qfname,portkeys,quants)
        now = time.time_ns()
        #Quantizer.aggregateBatchesH5(qfname,portkeys,quants,agrtype='kmeans') # agrtype can be 'quick' or 'kmeans'
        #print('Aggregation kmeans in %i ms'%((time.time_ns()-now)//1000000))
        #now = time.time_ns()
        Quantizer.aggregateBatchesH5(qfname,portkeys,quants,agrtype='quick') # agrtype can be 'quick' or 'kmeans'
        print('Aggregation quick in %i ms'%((time.time_ns()-now)//1000000))

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

