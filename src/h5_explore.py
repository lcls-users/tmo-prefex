import h5py
import psana
import psana
import matplotlib.pyplot as plt
import numpy as np
import os

def make hist

def main(run_files):
    run_folders_all = os.listdir(run_files)
    run_folders = []
    for file in run_folders_all:
        if "r008" in file:
            run_folders.append(file)
    
    for run in run_folders:
        hits_all = os.listdir(os.path.join(run_files, run))
        for hit in hits_all:
            if "config" not in hit:
                hit_file = hit
        
        data = h5py.File(os.path.join(run_files, run, hit), "r")
        
        print("data keys: ", data.keys())
        print("data[1] keys: ", data[list(data.keys())[0]].keys())
        print("data[1] keys: ", data[list(data.keys())[0]]['mrco_hsd'].keys())
        
        print(data[list(data.keys())[0]]['gmd'].keys())
        print(data[list(data.keys())[0]]['gmd']['events'])
        plt.plot(data[list(data.keys())[0]]['gmd']['events'][()])
        plt.show()
        print(data[list(data.keys())[0]]['gmd']['energy'])
        plt.plot(data[list(data.keys())[0]]['gmd']['energy'][()])
        plt.show()
        
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        # Plot each subplot
        axs[0].plot(data[list(data.keys())[0]]['gmd']['energy'][()], 'r')
        axs[0].set_title('GMD')

        axs[1].plot(data[list(data.keys())[0]]['xgmd']['energy'][()], 'b')
        axs[1].set_title('XGMD')

        # axs[2].plot(x, y3, 'purple')
        # axs[2].set_title('Exponential Function')

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()
        
        
        
        
        
        mrco_data = data[list(data.keys())[0]]['mrco_hsd']

        print(mrco_data.keys())
        for port in list(mrco_data.keys()):
            print(mrco_data[port].keys())
            # print(mrco_data[port]['tofs'])
            print(mrco_data[port]['nedges'][()])
            plt.plot(mrco_data[port]['nedges'][()])
            plt.show()
            
            tofs = mrco_data[port]['tofs'][()]
            # plt.plot(mrco_data[port]['tofs'][()])
            # plt.show()
            tof_dict = {}
            for tof in tofs:
                if tof in list(tof_dict.keys()):
                    tof_dict[tof] += 1
                else:
                    tof_dict[tof] = 1
                    
            total_data = np.zeros(max(list(tof_dict.keys())))
            
            for keys in tof_dict.keys():
                total_data[int(keys-1)] = tof_dict[keys]
            
            fig = plt.figure()       
            plt.plot(total_data)
            plt.show()
            
            # print(mrco_data[port]['addresses'])
            plt.plot(mrco_data[port]['slopes'][()], '+')
            plt.show()
            
            slopes = mrco_data[port]['slopes'][()]
            slopes_dict = {}
            for slope in slopes:
                if slope in list(slopes_dict.keys()):
                    slopes_dict[slope] += 1
                else:
                    slopes_dict[slope] = 1
                    
            slopes_data = np.zeros(max(list(slopes_dict.keys())))
            
            for keys in slopes_dict.keys():
                slopes_data[int(keys-1)] = slopes_dict[keys]
            
            fig = plt.figure()       
            plt.plot(slopes_data)
            plt.show()
            
            print("slopes: ", mrco_data[port]['slopes'][()])
            print("addresses: ", mrco_data[port]['addresses'][()])
            plt.plot(mrco_data[port]['addresses'][()], '+')
            plt.show()
            
            
            
            
            # data.close()
            # exit(1)
        data.close()
        exit(1)
            
        
    
    
    
    
if __name__ in "__main__":
    main(run_files="/sdf/data/lcls/ds/tmo/tmox1016823/scratch/coffee/h5files/")