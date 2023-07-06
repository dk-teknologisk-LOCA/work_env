import numpy as np
import scipy.signal as sg
import scipy.integrate as sint
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import glob
import os

global xsense_freq, trans_mat_xsens
xsense_freq = 100

#trans_mat_xsens_tool = np.vstack((np.hstack((R.from_euler('xyz', np.deg2rad([-90, 0, 0])).as_matrix(), np.array([[0.35391],[-0.71130],[0.02970]]))), [0, 0, 0, 1]))

def butter_lowpass_filter(data, cutoff, fs, order, nyq):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = sg.butter(order, normal_cutoff, btype='low', analog=False)
    y = sg.filtfilt(b, a, data)
    return y

def filtering(x):
    global xsense_freq
    # Filter requirements.
    #T = 5.0         # Sample Period
    fs = xsense_freq       # sample rate, Hz
    nyq = 0.5 * fs + 2      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hznyq = 0.5 * fs  # Nyquist Frequencyorder = 2       # sin wave can be approx represented as quadratic
    n = len(x) # total number of samples
    cutoff = 10
    order = 10

    return butter_lowpass_filter(x, cutoff, fs, order, nyq)

def recco(dir_name, x):
    title = os.path.basename(os.path.normpath(dir_name))[4:-4]
    print(title)
    #ts, fax, fay, faz for xsens, input vector in the prediction phase
    df = pd.read_csv(dir_name)
    data = pd.DataFrame(df)
    data = data[["ts", "fax", "fay", "faz"]]
    data = data.to_numpy()

    #aug_acc_vec = np.hstack((data[:,1:], np.ones((data.shape[0],1)))).T
    #data[:,1:] = np.matmul(trans_mat_xsens, aug_acc_vec)[:-1,:].T

    out = np.zeros((data.shape[0], 3))
    for i in range(1,data.shape[1]):
        out[:,i-1] = filtering(data[:,i])

    if x==1:
        #plot xsense data post filter
        fig0, axes0 = plt.subplots(3,1)
        fig0.suptitle("Acc " + title)
        axes0[0].plot(data[:,1]); axes0[0].set_ylabel("x"); axes0[0].plot(out[:,0]); axes0[0].set_ylabel("fx")
        axes0[1].plot(data[:,2]); axes0[1].set_ylabel("y"); axes0[1].plot(out[:,1]); axes0[1].set_ylabel("fy")
        axes0[2].plot(data[:,3]); axes0[2].set_ylabel("z"); axes0[2].plot(out[:,2]); axes0[2].set_ylabel("fz")


    vel = np.zeros(out.shape)
    for i in range(1, out.shape[0]):
        dt = 1/xsense_freq 
        vel[i] = vel[i-1] + out[i]*dt

    if x==1:
        #plot xsense data post filter
        fig1 = plt.figure()
        plt.title("Vel " + title)
        plt.plot(vel)
        plt.legend(["vx", "vy", "vz"])


    disp = np.zeros(vel.shape)
    #disp[0,:] = np.asarray([0.35391, -0.71130, 0.02970])
    for i in range(1, vel.shape[0]):
        dt = 1/xsense_freq 
        disp[i] = disp[i-1] + vel[i]*dt

    #rotmat = R.from_euler('yxz', np.deg2rad([90, 0, 0])).as_matrix()
    #disp = np.matmul(disp, rotmat)

    if x==1:
        #plot xsense data post filter
        fig1 = plt.figure()
        plt.title("Pos " + title)
        plt.plot(disp)
        plt.legend(["x", "y", "z"])

    fig2, axes2 = plt.subplots()
    plt.title("Pos 3D " + title)
    axes2.plot(disp[:,0], disp[:,1])
    axes2.scatter(disp[0,0], disp[0,1]) 
    axes2.set_ylabel("y=EAST")
    axes2.set_xlabel("x=NORTH")
    axes2.axis('equal')

cur_wd = os.getcwd()
print("Plot pre-data?")
#x = input()
x = 0
for dir in glob.glob(cur_wd+"/data_testing/sensor_fusion/prova_con_xsense/xsens_body_frame/xsense/*.csv"):
    print(dir)
    recco(dir, x)
    
plt.show()
