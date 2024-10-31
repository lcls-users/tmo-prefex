import psana
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import scipy

def peakfinder(window):
    grad = np.zeros(len(window))
    for count in range(len(window)-1):
        grad[count] = window[count+1] - window[count]
        
    grad = np.gradient(window)
    # grad = np.diff(window)
        
    
    # plt.plot(window)
    # plt.show()
    # zero_cross_plot = (window * grad)**4
    zero_cross_plot = grad
    
    # plt.plot(zero_cross_plot)
    # plt.show()
    
    zero_cross = []
    for count in range(len(zero_cross_plot)-1):
        if (zero_cross_plot[count] > 0 and zero_cross_plot[count + 1] < 0) or (zero_cross_plot[count] < 0 and zero_cross_plot[count + 1] > 0):
            if abs(zero_cross_plot[count]) < abs(zero_cross_plot[count+1]):
                zero_cross.append(count)
            else:
                zero_cross.append(count+1)
                                
    return zero_cross
                
        
        
        

def main(fname, fig_save_path, data_save_path, event_max):
    expname = os.getenv('expname')
    datapath = os.getenv('datapath')
    print("experiment name: ", expname)
    
    max_dict = {}
    
    ds = psana.DataSource(files = os.path.join(datapath, fname))
    run = next(ds.runs())
    det_name = list(run.detnames)[0]
    hsd = run.Detector(det_name)
    for i in range(event_max):
        
        if i % 500 == 0:
            print("event: ", i)
        try:
            evt = next(run.events())
        except:
            print("reached end at event: ", i)
            break
        if i == 0:
            chan = list(hsd.raw.peaks(evt).keys())[0]

        if hsd.raw.waveforms(evt):
            peaks = hsd.raw.peaks(evt)[chan][0]
            if len(peaks) != 2:
                print(f'for event {i}, peaks does not have 2 entries')
                continue
            if len(peaks[0]) != len(peaks[1]):
                print("peaks do not line up")
            if len(peaks[1]) > 2:
                #found peaks
                for p in range(1, len(peaks[1]) -1):
                    peak_locations = peakfinder(peaks[1][p])
                    for j in peak_locations:
                        peak_max = j + peaks[0][p] #max + initial offset
                        # print("offset: ", peaks[0][p], " peak pos: ", j, " peak max: ", peak_max)
                        if peak_max in list(max_dict.keys()):
                            max_dict[peak_max] += 1
                        else:
                            max_dict[peak_max] = 1

    
    
    
    with open(os.path.join(data_save_path, fname[:-5]), 'wb') as file:
        pickle.dump(max_dict, file)
    
    fig = plt.figure()       
    plt.bar(list(max_dict.keys()), list(max_dict.values()))
    plt.ylim(top=200)
    plt.xlim(left=4000, right=11000)
    fig.savefig(fig_save_path, dpi=fig.dpi)
    print("done")
    
    
    
    
if __name__ in "__main__":
    main(fname = "tmox1016823-r0084-s009-c000.xtc2", fig_save_path = "/sdf/home/b/bmencer/github/tmo-prefex/figures/fex_peaks_hist.png", data_save_path="/sdf/data/lcls/ds/tmo/tmox1016823/scratch/bmencer/peak_max", event_max=500000000)