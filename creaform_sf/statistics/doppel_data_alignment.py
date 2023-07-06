import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
from resample import *

def load_file_FANUC(filename):
    #loading data from CSV
    data = pd.read_csv(filename, delimiter=';', usecols=[1,2,3,4,5,6,7])
    df = pd.DataFrame(data)
    ts = data.index; ts = ts.to_numpy(); ts = ts.reshape(-1,1)
    
    vals = df.to_numpy()
    vals = np.hstack((ts, vals))
    
    return vals

def load_file_creaform(filename):
    #loading data from CSV
    data = pd.read_csv(filename, skiprows = 1, converters={'TZ': lambda x: str(x), 'RX': lambda x: str(x)}, usecols=[1,2,3,4,5,6,7,8])
    df = pd.DataFrame(data)
    ts = data.index; ts = ts.to_numpy(); ts = ts.reshape(-1,1)
    
    vals = df.to_numpy() #from the first column onwards because    
    vals = vals[:, :-1]

    for i in vals[:]:
        a = i[2]; b = i[3]
        i[2] = np.float64(a+b)   #unify the to columns for the z
        
    vals = np.delete(vals, 3, 1)   #eliminate the remaining excess column
    
    for i, value in enumerate(vals[:]):
        if np.fix(value[0]) == 0 and np.fix(value[1]) == 0:
            vals[i] = (vals[i-1]+vals[i+1])/2

    for i, val in enumerate(vals):
        if i<10:
            continue
        elif np.abs(vals[i-3][0]-vals[i+3][0]) > 2:
            break


    vals = vals[i:]
    ts = ts[i:]
    ts = ts - ts[0]
    vals = np.hstack((ts, vals))
    
    return vals

def rifasamento(ref, data):
    somma_gen = 10000
    best_tau = 0
    for tau in range(-100,100):
        delta_x = ref[int(np.fix(len(ref)/2)),1] - data[int(np.fix(len(ref)/2+tau)),1]
        delta_y = ref[int(np.fix(len(ref)/2)),2] - data[int(np.fix(len(ref)/2+tau)),2]
        delta_z = ref[int(np.fix(len(ref)/2)),3] - data[int(np.fix(len(ref)/2+tau)),3]
        somma_temp = np.power(delta_x, 2) + np.power(delta_y, 2) + np.power(delta_z, 2)

        if somma_temp < somma_gen:
            somma_gen = somma_temp
            best_tau = tau

    if best_tau != 0:
        if best_tau<0:
            temp = data[:,0].reshape(-1,1)
            data = np.roll(data[:,1:], -best_tau, axis=0)
            data = np.hstack((temp, data))
            data = data[best_tau:,:]
        if best_tau>0:
            temp = data[:,0].reshape(-1,1)
            data = np.roll(data[:,1:], -best_tau, axis=0)
            data = np.hstack((temp, data))
            data = data[:-best_tau,:]


    return data

def average(data):
    #prendi a lista, scomponila in elementi, crea un array 3d, e poi fai la average per le x, y e z e cosi via
    data_arr = data[0]
    for i in data[1:]:
        data_arr = np.dstack((data_arr, i))
    avg_list = []
    for i in range(data_arr.shape[1]):
        avg_list.append(np.divide(np.sum(data_arr[:, i, :], axis=1).reshape(-1,1), data_arr.shape[2]))
    avg_vec = np.array(avg_list).reshape(len(data_arr[0]),-1).T
    return avg_vec

def choose_dataset():
    x = input("Taastrup data: 1, Ringsted trials: 2. \n")
    if x == '1':
        foldername = "creaform_data/taastrup"
    elif x == '2':
        foldername = "creaform_data/ringsted"
    else:
        foldername = choose_dataset()

    return foldername

def c_or_f():
    x = input("Creaform: 1, FANUC: 2. \n")
    if x == '1':
        bot = 1
    elif x == '2':
        bot = 2
    else:
        bot = c_or_f()

    return bot

def plotter(gen_bins, total_histogram, delta_mean, delta_std, og_data_arr):
    common_bins = gen_bins
    tot_hist = total_histogram
    data_as_arr=og_data_arr
    positions = np.sqrt(np.power(data_as_arr[:,1,:], 2).astype(float)+np.power(data_as_arr[:,2,:], 2).astype(float)+np.power(data_as_arr[:,3,:], 2).astype(float))
    x = input("Show Boxplots?\n YES: 1\n NO: 2 \n")
    if x == '1':
        fig1 = plt.figure(1, figsize=plt.figaspect(0.5))
        plt.subplot(1,2,1)
        plt.hist(common_bins[:-1],common_bins, weights = tot_hist[0,:])
        plt.axvline(delta_mean[0], color='r', linestyle='dashed', linewidth=1, label="Mean = {}".format(delta_mean[0]))
        plt.axhline(delta_std[0], color='g', linestyle='solid', linewidth=1, label="Std = {}".format(delta_std[0]))
        plt.legend()
        plt.xlabel("$\Delta$ x [mm]")
        plt.ylabel("Counts")
        plt.subplot(1,2,2)
        plt.boxplot(data_as_arr[:,1,:].T)    
        fig2 = plt.figure(2, figsize=plt.figaspect(0.5)) 
        plt.subplot(1,2,1)
        plt.hist(common_bins[:-1],common_bins, weights = tot_hist[1,:])
        plt.axvline(delta_mean[1], color='r', linestyle='dashed', linewidth=1, label="Mean = {}".format(delta_mean[1]))
        plt.axhline(delta_std[1], color='g', linestyle='solid', linewidth=1, label="Std = {}".format(delta_std[1]))
        plt.legend()
        plt.xlabel("$\Delta$ y [mm]")
        plt.ylabel("Counts")
        plt.subplot(1,2,2)
        plt.boxplot(data_as_arr[:,2,:].T)   
        fig3 = plt.figure(3, figsize=plt.figaspect(0.5))
        plt.subplot(1,2,1)
        plt.hist(common_bins[:-1],common_bins, weights = tot_hist[2,:])
        plt.axvline(delta_mean[2], color='r', linestyle='dashed', linewidth=1, label="Mean = {}".format(delta_mean[2]))
        plt.axhline(delta_std[2], color='g', linestyle='solid', linewidth=1, label="Std = {}".format(delta_std[2]))
        plt.legend()
        plt.xlabel("$\Delta$ z [mm]")
        plt.ylabel("Counts")
        plt.subplot(1,2,2)
        plt.boxplot(data_as_arr[:,3,:].T)   
        fig4 = plt.figure(4, figsize=plt.figaspect(0.5))
        plt.subplot(1,2,1)
        plt.hist(common_bins[:-1],common_bins, weights = tot_hist[3,:])
        plt.axvline(delta_mean[3], color='r', linestyle='dashed', linewidth=1, label="Mean = {}".format(delta_mean[3]))
        plt.axhline(delta_std[3], color='g', linestyle='solid', linewidth=1, label="Std = {}".format(delta_std[3]))
        plt.legend()
        plt.xlabel("$\Delta$ P [mm]")
        plt.ylabel("Counts")
        plt.subplot(1,2,2)
        plt.boxplot(positions.T)   
        plt.show()
    elif x == '2':
        fig1 = plt.figure(1, figsize=plt.figaspect(0.5))
        plt.hist(common_bins[:-1],common_bins, weights = tot_hist[0,:])
        plt.axvline(delta_mean[0], color='r', linestyle='dashed', linewidth=1, label="Mean = {}".format(delta_mean[0]))
        plt.axhline(delta_std[0], color='g', linestyle='solid', linewidth=1, label="Std = {}".format(delta_std[0]))
        plt.legend()
        plt.xlabel("$\Delta$ x [mm]")
        plt.ylabel("Counts")  
        fig2 = plt.figure(2, figsize=plt.figaspect(0.5)) 
        plt.hist(common_bins[:-1],common_bins, weights = tot_hist[1,:])
        plt.axvline(delta_mean[1], color='r', linestyle='dashed', linewidth=1, label="Mean = {}".format(delta_mean[1]))
        plt.axhline(delta_std[1], color='g', linestyle='solid', linewidth=1, label="Std = {}".format(delta_std[1]))
        plt.legend()
        plt.xlabel("$\Delta$ y [mm]")
        plt.ylabel("Counts") 
        fig3 = plt.figure(3, figsize=plt.figaspect(0.5))
        plt.hist(common_bins[:-1],common_bins, weights = tot_hist[2,:])
        plt.axvline(delta_mean[2], color='r', linestyle='dashed', linewidth=1, label="Mean = {}".format(delta_mean[2]))
        plt.axhline(delta_std[2], color='g', linestyle='solid', linewidth=1, label="Std = {}".format(delta_std[2]))
        plt.legend()
        plt.xlabel("$\Delta$ z [mm]")
        plt.ylabel("Counts") 
        fig4 = plt.figure(4)
        plt.hist(common_bins[:-1],common_bins, weights = tot_hist[3,:])
        plt.axvline(delta_mean[3], color='r', linestyle='dashed', linewidth=1, label="Mean = {}".format(delta_mean[3]))
        plt.axhline(delta_std[3], color='g', linestyle='solid', linewidth=1, label="Std = {}".format(delta_std[3]))
        plt.legend()
        plt.xlabel("$\Delta$ P [mm]")
        plt.ylabel("Counts")
        plt.show()
    
if __name__ == "__main__":
    bot = c_or_f()
    if bot == 2:
        folder = "FANUC_data_Ringsted"
        file_list = glob.glob(os.path.join(folder,"*.txt"))
        tail_cut = 1
        scale_factor = 100
        ####GET ORIGINAL DATA
        og_data = []
        for file in file_list:
            fr = load_file_FANUC(file)
            og_data.append(fr)
        min_len = og_data[0].shape[0]
        for i in og_data:
            len_tmp = i.shape[0]
            if len_tmp<min_len:
                min_len=len_tmp

        for i, val in enumerate(og_data):
            og_data[i] = val[0:min_len][:]

        ####CREATE 3D ARRAY FOR THE ORIGINAL DATA FOR THE BOXPLOT
        data_as_arr = og_data[0]
        for i in og_data[1:]:
            data_as_arr = np.dstack((data_as_arr, i))

        delta_all = [[],[],[],[]] #list of all deltas, 3 elements (x,y,z) by N samples
        distances = []
        histos = []
        for i, run in enumerate(og_data):  
            data_temp = [x for j, x in enumerate(og_data) if j!=i]
            delta=run-average(data_temp) #delta between te current run and the average of all the others (I am subtracting all the 8 columns)
            
            ####Fill the list of all deltas
            delta_all[0].extend(delta[:,0].tolist()) 
            temp_histo_x, bin_edges_x = np.histogram(delta[:,0], range=(-tail_cut,tail_cut), bins=100) 
            
            delta_all[1].extend(delta[:,1].tolist()) 
            temp_histo_y, bin_edges_y = np.histogram(delta[:,1], range=(-tail_cut,tail_cut), bins=100) 
            
            delta_all[2].extend(delta[:,2].tolist()) 
            temp_histo_z, bin_edges_z = np.histogram(delta[:,2], range=(-tail_cut,tail_cut), bins=100) 

            #Create distance values, Pk=(x^2+y^2+z^2)^1/2 on the current run, Pavg = (x^2+y^2+z^2)^1/2 on the average(data_temp)
            P_k = np.sqrt(np.power(run[:,0],2).astype(float)+np.power(run[:,1],2).astype(float)+np.power(run[:,2],2).astype(float))
            P_avg = np.sqrt(np.power(average(data_temp)[:,0],2).astype(float)+np.power(average(data_temp)[:,1],2).astype(float)+np.power(average(data_temp)[:,2],2).astype(float))
            dist = P_k-P_avg
            delta_all[3].extend(dist.tolist())
            temp_histo_P, bin_edges_P = np.histogram(dist, range=(-tail_cut,tail_cut), bins=100) 

            #Fill the histogram list
            histos.append([[temp_histo_x.T, bin_edges_x.T], [temp_histo_y.T, bin_edges_y.T], [temp_histo_z.T, bin_edges_z.T],[temp_histo_P.T, bin_edges_P.T]]) #store the frequency values and the values corresponding to those frequencies as sublists in a list


        delta_all_core = [[],[],[],[]]
        for i in range(4):
            for j in delta_all[i]:  
                if j<-tail_cut or j>tail_cut:
                    continue
                delta_all_core[i].append(j)
        
        delta_mean = [[np.mean(np.array(delta_all_core[0], dtype=object))], [np.mean(np.array(delta_all_core[1], dtype=object))], [np.mean(np.array(delta_all_core[2], dtype=object))], [np.mean(np.array(delta_all_core[3], dtype=object))]]
        delta_std = [[np.std(np.array(delta_all_core[0], dtype=object))],[np.std(np.array(delta_all_core[1], dtype=object))],[np.std(np.array(delta_all_core[2], dtype=object))],[np.std(np.array(delta_all_core[3], dtype=object))]]

        print("mean and std of the deltas", delta_mean, delta_std)

        #HISTOGRAM WITH THE AVERAGE OF THE VALUES
        tot_hist = np.zeros((len(histos[0][0][0]), len(histos[0]))).T
        for i, histo in enumerate(histos):
            tot_hist[0,:] = tot_hist[0,:] + histo[0][0]
            tot_hist[1,:] = tot_hist[1,:] + histo[1][0]
            tot_hist[2,:] = tot_hist[2,:] + histo[2][0]
            tot_hist[3,:] = tot_hist[3,:] + histo[3][0]
        tot_hist = tot_hist/scale_factor
        common_bins = histos[0][0][1]

        plotter(common_bins, tot_hist, delta_mean, delta_std, data_as_arr)

    elif bot == 1:
        n_sample = 1000
        scale_factor = 100
        tail_cut = 5

        folder = choose_dataset()
        file_list = glob.glob(os.path.join(folder,"*.CSV"))

        ####GET ORIGINAL DATA AND RESAMPLED DATA
        data = []
        og_data = []
        first_run = load_file_creaform(file_list[0])
        for file in file_list[1:]:
            fr = load_file_creaform(file)
            og_data.append(fr)
            rs_fr = resample(fr, n_sample)
            #rf_rs_fr = rifasamento(rs_fr)
            #data.append(rf_rs_fr)
            data.append(rs_fr)
        
        ####FIND WHICH RUN HAS THE MINIMUM NUMBER OF SAMPLES THEN CUT ALL THE RUNS TO THAT LENGTH, BOTH FOR THE RESAMPLED AND THE ORIGINAL DATA
        min_len = data[0].shape[0]
        for i in data:
            len_tmp = i.shape[0]
            if len_tmp<min_len:
                min_len=len_tmp

        for i, val in enumerate(data):
            data[i] = val[0:min_len][:]
        
        min_len = og_data[0].shape[0]
        for i in og_data:
            len_tmp = i.shape[0]
            if len_tmp<min_len:
                min_len=len_tmp

        for i, val in enumerate(og_data):
            og_data[i] = val[0:min_len][:]

        ####CREATE 3D ARRAY FOR THE ORIGINAL DATA FOR THE BOXPLOT
        data_as_arr = og_data[0]
        for i in og_data[1:]:
            data_as_arr = np.dstack((data_as_arr, i))

        delta_all = [[],[],[],[]] #list of all deltas, 3 elements (x,y,z) by N samples
        distances = []
        histos = []
        for i, run in enumerate(data):  
            data_temp = [x for j, x in enumerate(data) if j!=i]
            delta=run-average(data_temp) #delta between te current run and the average of all the others (I am subtracting all the 8 columns)
            
            ####Fill the list of all deltas
            delta_all[0].extend(delta[:,0].tolist()) 
            temp_histo_x, bin_edges_x = np.histogram(delta[:,0], range=(-tail_cut,tail_cut), bins=100) 
            
            delta_all[1].extend(delta[:,1].tolist()) 
            temp_histo_y, bin_edges_y = np.histogram(delta[:,1], range=(-tail_cut,tail_cut), bins=100) 
            
            delta_all[2].extend(delta[:,2].tolist()) 
            temp_histo_z, bin_edges_z = np.histogram(delta[:,2], range=(-tail_cut,tail_cut), bins=100) 

            #Create distance values, Pk=(x^2+y^2+z^2)^1/2 on the current run, Pavg = (x^2+y^2+z^2)^1/2 on the average(data_temp)
            P_k = np.sqrt(np.power(run[:,0],2).astype(float)+np.power(run[:,1],2).astype(float)+np.power(run[:,2],2).astype(float))
            P_avg = np.sqrt(np.power(average(data_temp)[:,0],2).astype(float)+np.power(average(data_temp)[:,1],2).astype(float)+np.power(average(data_temp)[:,2],2).astype(float))
            dist = P_k-P_avg
            delta_all[3].extend(dist.tolist())
            temp_histo_P, bin_edges_P = np.histogram(dist, range=(-tail_cut,tail_cut), bins=100) 

            #Fill the histogram list
            histos.append([[temp_histo_x.T, bin_edges_x.T], [temp_histo_y.T, bin_edges_y.T], [temp_histo_z.T, bin_edges_z.T],[temp_histo_P.T, bin_edges_P.T]]) #store the frequency values and the values corresponding to those frequencies as sublists in a list


        delta_all_core = [[],[],[],[]]
        for i in range(4):
            for j in delta_all[i]:  
                if j<-tail_cut or j>tail_cut:
                    continue
                delta_all_core[i].append(j)
        
        delta_mean = [[np.mean(np.array(delta_all_core[0], dtype=object))], [np.mean(np.array(delta_all_core[1], dtype=object))], [np.mean(np.array(delta_all_core[2], dtype=object))], [np.mean(np.array(delta_all_core[3], dtype=object))]]
        delta_std = [[np.std(np.array(delta_all_core[0], dtype=object))],[np.std(np.array(delta_all_core[1], dtype=object))],[np.std(np.array(delta_all_core[2], dtype=object))],[np.std(np.array(delta_all_core[3], dtype=object))]]

        print("mean and std of the deltas", delta_mean, delta_std)

        #HISTOGRAM WITH THE AVERAGE OF THE VALUES
        tot_hist = np.zeros((len(histos[0][0][0]), len(histos[0]))).T
        for i, histo in enumerate(histos):
            tot_hist[0,:] = tot_hist[0,:] + histo[0][0]
            tot_hist[1,:] = tot_hist[1,:] + histo[1][0]
            tot_hist[2,:] = tot_hist[2,:] + histo[2][0]
            tot_hist[3,:] = tot_hist[3,:] + histo[3][0]
        tot_hist = tot_hist/scale_factor
        common_bins = histos[0][0][1]

        plotter(common_bins, tot_hist, delta_mean, delta_std, data_as_arr)

    