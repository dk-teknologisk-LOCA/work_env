###########Compare test runs############
#We do not have yet the type of data, so for now i will assume the data to be an x by 6 by y matrix, where 6 is (x,y,z,R,P,Y), x is the number of data 
# observations and y is the number of test runs.

#The file takes N runs take frm the same sensor. Creates a 3D matrix with M number of samples (rows), 6 columns (x,y,z, Roll, Pitch, Yaw) and Q 
# slices (as many as the test runs). Then, it slices this 3D matrix by column, comparing the xs from every run, the ys from every run and so on.
# Then it uses this data to calculate the statistic on every sample by x, y, z and so on. This returns, then, 3 final matrixes with a N samples (rows) and 
# 6 dimensions (columns) structure composed of the statistics on the rus and observations and samples.

import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
import pandas as pd
import os
import glob
import csv 

def load_run_data(filepath):
    #loading data from Excel
    data = pd.read_excel(filepath)
    df = pd.DataFrame(data)
    vals = df.to_numpy()

    return vals

def load_all_runs(foldername):
    file_path = foldername
    print("file path", file_path)
    file_list = glob.glob(os.path.join(file_path,"*.ods"))
    print("file list", file_list)

    vals_tot = load_run_data(file_list[0])
    del file_list[0]
    
    for file in file_list:
        vals_new = load_run_data(file)

        vals_tot = np.dstack((vals_tot, vals_new))

    df_list = []
    for i in range(6):
        df_list.append(vals_tot[:,i,:])

    return df_list

def statistics(data): #data expected as a (1000, 3) matrix
    data = data
    avg = np.mean(data,1)
    std = np.std(data,1)
    sem = scipy.stats.sem(data,1)
    
    return avg, std, sem

def all_stats(foldername):
    df_list = load_all_runs(foldername)
    avg_x, std_x, sem_x = statistics(df_list[0])
    avg_y, std_y, sem_y = statistics(df_list[1])
    avg_z, std_z, sem_z = statistics(df_list[2])
    avg_roll, std_roll, sem_roll = statistics(df_list[3])
    avg_pitch, std_pitch, sem_pitch = statistics(df_list[4])
    avg_yaw, std_yaw, sem_yaw = statistics(df_list[5])

    avg_vec_data = np.array([avg_x, avg_y, avg_z, avg_roll, avg_pitch, avg_yaw])
    std_vec = np.array([std_x, std_y, std_z, std_roll, std_pitch, std_yaw])
    sem_vec = np.array([sem_x, sem_y, sem_z, sem_roll, sem_pitch, sem_yaw])
    
    plt.figure()
    plt.subplot(311)
    plt.ylabel("x")
    plt.plot(avg_x)
    plt.subplot(312)
    plt.ylabel("y")
    plt.plot(avg_y)
    plt.subplot(313)
    plt.ylabel("z")
    plt.plot(avg_z)
    
    avg_vec_data = np.transpose(avg_vec_data)
    std_vec = np.transpose(std_vec)
    sem_vec = np.transpose(sem_vec)

    return avg_vec_data, std_vec, sem_vec

if __name__=="__main__":
    #obs = 1000
    #runs = 3
    #data_vec = np.zeros([obs, 6, runs]) #Possible jumps from, e.g., 30^ to 390^?
    foldername = "/home/lorenzo/data_testing/source/"
    avg_vec, std_vec, sem_vec = all_stats(foldername)
    
    avg_vec = np.insert(avg_vec, 0, np.zeros(6), axis = 0)
    print(avg_vec)
    np.savetxt("avg_run.csv", avg_vec, delimiter=",")
    #the avg_vec can then be taken to form the basis of the comparison with the 
    #actual data from the robot movement