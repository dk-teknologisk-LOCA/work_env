from random import randrange
import pandas as pd
import pathlib
from filterpy.kalman import ExtendedKalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints , JulierSigmaPoints
import matplotlib.pyplot as plt
from numpy import array, arange, asarray, sqrt, eye, diag, linalg, ones, transpose, subtract, pi, sin, matmul, vstack, hstack, loadtxt, mean, linspace, polyfit, cos
from numpy.random import randn, normal, uniform
from numpy.polynomial import Chebyshev
from scipy.optimize import curve_fit
import scipy.signal as sg
from sklearn.svm import OneClassSVM

#UKF with acceleration smoothing from past data
dt= 1/80
def read_data_creaform():
    filepath = pathlib.Path(__file__).parent / "data/creaform/31_03_23.CSV"
    df = pd.read_csv(filepath, skiprows=1, usecols=[0,1,2,3,7], header=0, names=['TS', 'TX', 'TY', 'TZ', 'Valid'])
    full_pos = pd.DataFrame(df); full_pos[['TX', 'TY', 'TZ']] = full_pos[['TX', 'TY', 'TZ']]/1000.
    he_calib_mat = loadtxt(pathlib.Path(__file__).parent.parent / "hand_eye_calibration/he_calibration_matrix.txt")    
    full_pos[['TX', 'TY', 'TZ']] = matmul(he_calib_mat, hstack((full_pos[['TX', 'TY', 'TZ']].to_numpy(), ones(full_pos["TS"].to_numpy().shape).reshape(-1,1))).T).T[:,:-1]

    return full_pos

def read_true_data():
    #TX, TY, TZ for the robot
    filename = pathlib.Path(__file__).parent / "data/ROS_files/run_1/run_1_tf.csv"
    data = pd.DataFrame(pd.read_csv(filename))
    data = data[["parent_frame", "child_frame", "posx", "posy", "posz"]]
    data = data[["posx", "posy", "posz"]].to_numpy()

    return data

def plotit(true_data, creaform_data=None, recon_pos=None, recon_acc=None):
    dims = ["x", "y", "z"]
    for i in range(len(dims)):
        fig, ax = plt.subplots(2)
        fig.suptitle(dims[i])
        #ax[0].plot(recon_pos[:,i], label = "recon pos")
        ax[0].plot(creaform_data[:,i], '+', label = "crea pos")
        ax[0].grid()
        ax0_bis = ax[0].twiny()
        ax0_bis.plot(true_data[:,i], 'r--', label = "UR pos")
        #ax[1].plot(recon_acc[:,i], label = "recon acc")
        #ax[1].grid()
        #ax[2].plot(true_data-recon_data)
        plt.legend(loc=2)

def butter_lowpass_filter(data, cutoff, fs, order, nyq):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = sg.butter(order, normal_cutoff, btype='low', analog=False)
    y = sg.filtfilt(b, a, data)
    return y

def filtering(x):
    # Filter requirements.
    #T = 5.0         # Sample Period
    fs = 80.       # sample rate, Hz
    nyq = 0.5 * fs + 2      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hznyq = 0.5 * fs  # Nyquist Frequencyorder = 2       # sin wave can be approx represented as quadratic
    n = len(x) # total number of samples
    cutoff = 10
    order = 10
    return butter_lowpass_filter(x, cutoff, fs, order, nyq)

#data
pos= read_data_creaform()
print(pos)  
posx = pos[["TS", "TX", "Valid"]]
posy = pos[["TS", "TY", "Valid"]]
posz = pos[["TS", "TZ", "Valid"]]
#remove unvalid points
posx = posx[posx["Valid"] != 0]
#Filter wrong points
"""for i in range(30, len(posx.to_numpy())-40):
    point_dist = []
    for  j in range(i-4, i+4):
        point_dist.append(sqrt((posx["TX"].iloc[j]-posx["TX"].iloc[j-1])**2. + (posx["TS"].iloc[j]-posx["TS"].iloc[j-1])**2.))
    avg_dist = mean(point_dist)
    #print("avg dist", avg_dist)
    point_dist = sqrt((posx["TX"].iloc[i]-posx["TX"].iloc[i-1])**2. + (posx["TS"].iloc[i]-posx["TS"].iloc[i-1])**2.)
    #print("point dist", point_dist)
    if point_dist > avg_dist*2.:
        posx["TX"].iloc[i] = 0
        posx["Valid"].iloc[i] = 0
    else:
        continue
posx = posx[posx["Valid"] != 0]"""

print(posx)
posxfilt = filtering(posx["TX"])
plt.plot(posx["TS"], posxfilt, ".")
plt.plot(posx["TS"], posx["TX"], "*")
plt.show()

true_data = read_true_data()

fig, ax = plt.subplots()
ax.plot(posx[:,0], fitted_data, '.', label = "fitted pos")
ax.plot(posx[:,0], posx[:,1], "+", label = "crea pos")
plt.legend()
ax.grid()
ax0_bis = ax.twiny()
ax0_bis.plot(true_data[:,0], 'r--', label = "UR pos")
plt.legend(loc=2)
plt.show()