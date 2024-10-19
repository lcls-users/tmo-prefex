#!/cds/sw/ds/ana/conda2/inst/envs/ps-4.5.7-py39/bin/python3

import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys
import re

verbose = True

def main(fname):
    with h5py.File(fname,'r') as q:
        if verbose: 
            print(q.keys())
        portkeys = [k for k in q.keys() if re.search('port',k)]
        if verbose: 
            _=[print(k) for k in portkeys]
        labellist = []
        portkeys = ['port_13','port_12','port_0','port_15']
        plt.figure(figsize=(12,8))
        i = 0
        for k in portkeys:
            plotk = False
            h = np.sum(q[k]['hist'][()],axis=1)
            qb = q[k]['qbins'][()]
            qdiff = qb[1:]-qb[:-1]
            print(np.min(qdiff))
            if re.search('port_15',k):
                plotk=True
                labellist += ['horiz (north)']
            if re.search('port_0',k):
                plotk=True
                labellist += ['vert']
            if re.search('port_12',k):
                plotk=True
                labellist += ['horiz (south)']
            if re.search('port_13',k):
                plotk=True
                labellist += ['\'13\'']
            if plotk:
                plt.stairs((i*100.)+h/qdiff,(qb-qb[0])/8/6)
                i += 1
        plt.xlabel('ToF [ns]')
        #plt.xlim(40,60)
        plt.xlim(20,120)
        plt.legend(labellist)
        plt.tight_layout()
        plt.savefig('./figures/qbinsRecovered_13_12_0_4.png')
        plt.show()
        portkeys = ['port_13','port_5','port_12','port_4','port_0','port_1','port_15','port_14']
        rets = [0,25,150,175,325,350,425,450]
        fig,axs = plt.subplots(1,4)
        fig.set_figwidth(10)
        fig.set_figheight(5)
        for i in range(0,len(portkeys),2):
            label='%s, ret = -%iV'%(portkeys[i],rets[i])
            axs[i//2].pcolor(q[portkeys[i//2]]['hist'][()][:,::32].T,cmap='Greys',vmin=0,vmax=3)
            axs[i//2].set_title(label)
            axs[i//2].set_xlabel('qbins')
        axs[0].set_ylabel('shot number')
        plt.savefig('./figures/qbinsSnow_ports_13_12_0_15.png')
        plt.show()

        '''
        N2O:
        0V retardation on port 13.
        25V retardation on port 5.
        150V retardation on port 12.
        175V retardation on port 4.
        325V retardation on port 0.
        350V retardation on port 1.
        425V retardation on port 15.
        450V retardation on port 14.
        '''
        labellist = []
        scale=1.25
        plt.figure(figsize=(10,5))
        for i,k in enumerate(portkeys):
            if i >2:
                scale=3
            if i >5:
                scale=10
            h = np.sum(q[k]['hist'][()],axis=1)
            qb = q[k]['qbins'][()]
            qdiff = qb[1:]-qb[:-1]
            plt.stairs((i*100.)+(scale)*h/qdiff,(qb-qb[0])/8/6,lw=2,label='%s, ret = -%iV'%(k,rets[i]))
            labellist += ['%s,ret=%i'%(k,rets[i])]
        plt.plot([29.5,29.5,30,30],[50,80,80,100],'-',color='k',label = '')
        plt.plot([30,30,34,34],[150,180,180,200],'-',color='k')
        plt.plot([34,34,35,35],[250,280,280,300],'-',color='k')
        plt.plot([35,35,45,45],[350,380,380,400],'-',color='k')
        plt.plot([45,45,49,49],[450,480,480,500],'-',color='k')
        plt.plot([49,49,64,64],[550,580,580,600],'-',color='k')
        plt.plot([64,64,75,75],[650,680,680,700],'-',color='k')
        plt.plot([35,35,36,36],[50,70,70,100],'-',color='k')
        plt.plot([36,36,43,43],[150,170,170,200],'-',color='k')
        plt.plot([43,43,45,45],[250,270,270,300],'-',color='k')
        plt.plot([45,45,80,80],[350,370,370,400],'-',color='k')
        plt.plot([48.5,48.5,51,51],[75,90,90,100],'-',color='k')
        plt.plot([51,51,98,98],[175,190,190,200],'-',color='k')
        plt.plot([90,90,113,113],[85,90,90,100],'-',color='k')
        plt.plot([27.5,27.5,28,28],[25,90,90,100],'-',color='k')
        plt.plot([28,28,30.5,30.5],[125,190,190,200],'-',color='k')
        plt.plot([30.5,30.5,31,31],[225,290,290,300],'-',color='k')
        plt.plot([31,31,37.5,37.5],[325,390,390,400],'-',color='k')
        plt.plot([37,37,39,39],[425,490,490,500],'-',color='k')
        plt.plot([39,39,43,43],[525,590,590,600],'-',color='k')
        plt.plot([43,43,50,50],[625,690,690,700],'-',color='k')
        plt.xlabel('ToF [ns]')
        plt.ylabel('Signal [arb]')
        plt.xlim(20,120)
        plt.tight_layout()
        plt.legend()
        plt.savefig('./figures/tofsRecovered_retorder.png')
        plt.show()

        labellist = []
        scale=1.25
        plt.figure(figsize=(10,5))
        for i,k in enumerate(portkeys):
            if i >2:
                scale=3
            if i >5:
                scale=10
            h = np.sum(q[k]['hist'][()],axis=1)
            qb = q[k]['qbins'][()]
            qdiff = qb[1:]-qb[:-1]
            plt.stairs((i*100.)+(scale)*h/qdiff,np.arange(len(qb)),lw=2,label='%s, ret = -%iV'%(k,rets[i]))
            labellist += ['%s,ret=%i'%(k,rets[i])]
        plt.xlabel('qbins [indx]')
        plt.ylabel('Cnts/qwidth [arb]')
        plt.xlim(0,64)
        plt.tight_layout()
        plt.legend()
        plt.savefig('./figures/qbinsRecovered_retorder.png')
        plt.show()

    return


if __name__ == '__main__':
    if len(sys.argv)<2:
        print('./src/plotCompareQuantNonquant.py <quantizename>')
    else:
        main(sys.argv[1])

