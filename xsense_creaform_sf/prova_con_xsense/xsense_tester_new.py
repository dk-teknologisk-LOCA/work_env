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

def rotate_to_ENU(data_point):
    return R.from_quat(data_point[3:]).apply(data_point[:3])

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
    cutoff = 30
    order = 20

    return butter_lowpass_filter(x, cutoff, fs, order, nyq)

def recco(dir_name, x):
    plot_max = -1
    title = os.path.basename(os.path.normpath(dir_name))[4:-4]
    print(title)
    #ts, fax, fay, faz for xsens, input vector in the prediction phase
    df = pd.read_csv(dir_name)
    df = pd.DataFrame(df)
    df = df[["ts", "fax", "fay", "faz", "ox", "oy", "oz", "ow"]]
    #print("data\n", df)
    data = df.to_numpy()

    #calculate initial orientation of the ENU wrt to W
    #take first reading of the orientation in order to do that
    init_orientation = R.from_quat(data[0, 4:]).inv()

    #rotate the single points
    rotated_data = np.zeros(data[:,:4].shape); rotated_data[:,0] = df["ts"]
    for i in range(len(data[:,0])):
        rotated_data[i,1:] = rotate_to_ENU(data[i, 1:])
    
    #apply rotation towards the World Frame
    data_in_W = np.hstack((rotated_data[:,0].reshape(-1,1), init_orientation.apply(rotated_data[:,1:])))
    out = np.zeros(data_in_W.shape); out[:, 0] =df["ts"]
    for i in range(1,data_in_W.shape[1]):
        out[:,i] = filtering(data_in_W[:,i])

    vel = np.zeros(out.shape)
    for i in range(out.shape[0]):
        dt = 1/xsense_freq 
        vel[i,1:] = vel[i-1, 1:] + out[i, 1:]*dt

    disp = np.zeros(vel.shape)
    disp[0,1:] = np.asarray([0.35391, -0.71130, 0.02970]) #initial position
    for i in range(1, vel.shape[0]):
        dt = 1/xsense_freq 
        disp[i, 1:] = disp[i-1, 1:] + vel[i, 1:]*dt

    #rotmat = R.from_euler('yxz', np.deg2rad([90, 0, 0])).as_matrix()
    #disp = np.matmul(disp, rotmat)

    if x==1:
        #plot xsense data post filter
        fig0, axes0 = plt.subplots(3,1)
        fig0.suptitle("Acc " + title)
        axes0[0].plot(data_in_W[:plot_max,1]); axes0[0].set_ylabel("x"); axes0[0].plot(out[:plot_max,1]); axes0[0].set_ylabel("fx")
        axes0[1].plot(data_in_W[:plot_max,2]); axes0[1].set_ylabel("y"); axes0[1].plot(out[:plot_max,2]); axes0[1].set_ylabel("fy")
        axes0[2].plot(data_in_W[:plot_max,3]); axes0[2].set_ylabel("z"); axes0[2].plot(out[:plot_max,3]); axes0[2].set_ylabel("fz")

        #plot xsense data post filter
        fig1 = plt.figure()
        plt.title("Vel " + title)
        plt.plot(vel[:plot_max,1:])
        plt.legend(["vx", "vy", "vz"])


        #plot xsense data post filter
        fig1 = plt.figure()
        plt.title("Pos " + title)
        plt.plot(disp[:plot_max,1:])
        plt.legend(["x", "y", "z"])

        fig2, axes2 = plt.subplots()
        plt.title("Pos 2D " + title)
        axes2.plot(disp[:plot_max,1], disp[:plot_max,2])
        axes2.scatter(disp[0,1], disp[0,2]) 
        axes2.set_ylabel("y=EAST")
        axes2.set_xlabel("x=NORTH")
        axes2.axis('equal')

    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    axes.plot3D(disp[:plot_max, 1], disp[:plot_max,2], disp[:plot_max, 3]); axes.scatter(disp[0, 1], disp[0,2], disp[0, 3])
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_zlabel("z")

cur_wd = os.getcwd()
print("Plot pre-data?"); #x = input()
x = 1
for dir in glob.glob(cur_wd+"/data_testing/sensor_fusion/prova_con_xsense/xsense/*.csv"):
    print(dir)
    recco(dir, x)
    
plt.show()
