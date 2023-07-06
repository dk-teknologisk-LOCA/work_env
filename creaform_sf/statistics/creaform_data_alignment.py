import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
from resample import *

def load_file(filename):
    #loading data from CSV
    data = pd.read_csv(filename, skiprows = 1, converters={'TZ': lambda x: str(x), 'RX': lambda x: str(x)}, usecols=[1,2,3,4,5,6,7,8])
    df = pd.DataFrame(data)
    ts = data.index; ts = ts.to_numpy(); ts = ts.reshape(-1,1)
    
    vals = df.to_numpy() #from the first column onwards because    
    
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
        elif np.abs(vals[i-3][0]-vals[i+3][0]) > 1:
            break


    vals = vals[i:, 0:5]
    ts = ts[i:]
    ts = ts - ts[0]
    #vals = np.hstack((ts, vals))
    

    return vals, ts

def rifasamento(ref, data):
    somma_gen = 10000
    best_tau = 0
    for tau in range(-150,150):
        #tau = tau/2
        #delta_y = abs(ref[int(np.fix(len(ref)/2)),2]) - abs(data[int(np.fix(len(ref)/2+tau)),2]) #(|x1|-|x2|)^2, vectorial
        #delta_z = abs(ref[int(np.fix(len(ref)/2)),3]) - abs(data[int(np.fix(len(ref)/2+tau)),3])
        #delta_x = abs(ref[int(np.fix(len(ref)/2)),1]) - abs(data[int(np.fix(len(ref)/2+tau)),1])
        #somma_temp = np.power(delta_x,2)+np.power(delta_y,2)+np.power(delta_z,2)
        #delta_y = np.power(ref[int(np.fix(len(ref)/2)),2],2) - np.power(data[int(np.fix(len(ref)/2+tau)),2],2)
        #delta_z = np.power(ref[int(np.fix(len(ref)/2)),3],2) - np.power(data[int(np.fix(len(ref)/2+tau)),3],2)
        #delta_x = np.power(ref[int(np.fix(len(ref)/2)),1],2) - np.power(data[int(np.fix(len(ref)/2+tau)),1],2)
        #somma_temp = delta_x+delta_y+delta_z
        #delta_y = ref[int(np.fix(len(ref)/2)),2] - data[int(np.fix(len(ref)/2+tau)),2]
        #delta_z = ref[int(np.fix(len(ref)/2)),3] - data[int(np.fix(len(ref)/2+tau)),3]
        #delta_x = ref[int(np.fix(len(ref)/2)),1] - data[int(np.fix(len(ref)/2+tau)),1]
        #somma_temp = np.power(delta_x,2) + np.power(delta_y, 2) + np.power(delta_z,2)
        #delta_y = ref[int(np.fix(len(ref)/2)),2] - data[int(np.fix(len(ref)/2+tau)),2]
        #delta_z = ref[int(np.fix(len(ref)/2)),3] - data[int(np.fix(len(ref)/2+tau)),3]
        #delta_x = ref[int(np.fix(len(ref)/2)),1] - data[int(np.fix(len(ref)/2+tau)),1]
        #somma_temp = np.sqrt(np.power(delta_x,2) + np.power(delta_y, 2) + np.power(delta_z,2))

        delta_x = ref[int(np.fix(len(ref)/2)),1] - data[int(np.fix(len(ref)/2+tau)),1]
        delta_y = ref[int(np.fix(len(ref)/2)),2] - data[int(np.fix(len(ref)/2+tau)),2]
        delta_z = ref[int(np.fix(len(ref)/2)),3] - data[int(np.fix(len(ref)/2+tau)),3]
        somma_temp = np.power(delta_x, 2) + np.power(delta_y, 2) + np.power(delta_z, 2)

        if somma_temp < somma_gen:
            somma_gen = somma_temp
            best_tau = tau
    #print("best tau", best_tau)
            #best_idx=int(np.fix(len(ref)/2+tau))
            
    #print('best_idx %s'%best_idx)
    #print('translated index %s' %(int(np.fix(len(ref)/2+best_tau))))
    if best_tau != 0:
        if best_tau<0:
            #temp = data[:,0].reshape(-1,1)
            data = np.roll(data, -best_tau, axis=0)
            #data = np.hstack((temp, data))
            data = data[(-best_tau):,:]
        if best_tau>0:
            #temp = data[:,0].reshape(-1,1)
            data = np.roll(data, -best_tau, axis=0)
            #data = np.hstack((temp, data))
            data = data[:(-best_tau),:]


    return data#, best_idx

if __name__ == "__main__":
    n_sample = 3000
    x = input("Taastrup data: 1, Ringsted trials: 2. \n")
    if x == '1':
        foldername = "creaform_data/taastrup"
    elif x == '2':
        foldername = "creaform_data/ringsted"
    file_list = glob.glob(os.path.join(foldername,"*.CSV"))
    #print(file_list[0])
    valsref, tsref = load_file(file_list[0])
    print("pre resample", valsref.shape)
    res_valsref = resample(valsref, n_sample)
    print("post resample", res_valsref.shape)

    fig1 = plt.figure()
    ax11 = fig1.add_subplot(3,1,1)
    ax21 = fig1.add_subplot(3,1,2)
    ax31 = fig1.add_subplot(3,1,3)
    fig2 = plt.figure()
    ax12 = fig2.add_subplot(3,1,1)
    ax22 = fig2.add_subplot(3,1,2)
    ax32 = fig2.add_subplot(3,1,3)
    ax11.plot(res_valsref[:,0])
    ax21.plot(res_valsref[:,1])
    ax31.plot(res_valsref[:,2])
    ax12.plot(res_valsref[:,0], linestyle="dashed")
    ax12.legend("ref")
    ax22.plot(res_valsref[:,1])
    ax32.plot(res_valsref[:,2])
    for file in file_list[1:]:
        #print(file)
        vals, ts = load_file(file) 
        print("pre resample", valsref.shape)
        res_vals = resample(vals, n_sample)
        print("post resample", res_valsref.shape)
        ax11.plot(res_vals[:,0])
        ax21.plot(res_vals[:,1])
        ax31.plot(res_vals[:,2])
        rif_vals = rifasamento(res_valsref, res_vals)

        ax12.plot(rif_vals[:,0])
        ax22.plot(rif_vals[:,1])
        ax32.plot(rif_vals[:,2])

    plt.show()