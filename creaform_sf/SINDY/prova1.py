import pysindy as ps
import matplotlib.pyplot as plt
import pathlib
from numpy import *
import pandas as pd
import scipy.signal as sg

def read_data_creaform():
    filepath = pathlib.Path(__file__).parent.parent / "data/creaform/31_03_23.CSV"
    df = pd.read_csv(filepath, skiprows=1, usecols=[0,1,2,3,7], header=0, names=['TS', 'TX', 'TY', 'TZ', 'Valid'])
    full_pos = pd.DataFrame(df); full_pos[['TX', 'TY', 'TZ']] = full_pos[['TX', 'TY', 'TZ']]/1000.
    he_calib_mat = loadtxt(pathlib.Path(__file__).parent.parent.parent / "hand_eye_calibration/he_calibration_matrix.txt")    
    full_pos[['TX', 'TY', 'TZ']] = matmul(he_calib_mat, hstack((full_pos[['TX', 'TY', 'TZ']].to_numpy(), ones(full_pos["TS"].to_numpy().shape).reshape(-1,1))).T).T[:,:-1]

    return full_pos

def read_true_data():
    #TX, TY, TZ for the robot
    filename = pathlib.Path(__file__).parent.parent / "data/ROS_files/run_1/run_1_tf.csv"
    data = pd.DataFrame(pd.read_csv(filename))
    data = data[["parent_frame", "child_frame", "posx", "posy", "posz"]]
    data = data[["posx", "posy", "posz"]].to_numpy()

    return data

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

#data
pos= read_data_creaform()
posx = pos[["TS", "TX", "Valid"]]; posy = pos[["TS", "TY", "Valid"]]; posz = pos[["TS", "TZ", "Valid"]]

#move unvalid points
#identify gap areas
gaps = posx["Valid"] == 0






posxfilt = filtering(posx["TX"])
xdata = posx[["TS", "TX"]]
xdata.set_index("TS", inplace=True)
xdata["TX"]  = posxfilt
model = ps.SINDy(feature_names=xdata.columns)
model.fit(xdata.values, t = xdata.index.values)
model.print()