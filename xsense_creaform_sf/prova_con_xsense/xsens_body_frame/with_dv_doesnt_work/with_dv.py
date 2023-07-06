import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import scipy.signal as sg

global xsense_freq
xsense_freq = 100

#ts, fax, fay, faz for xsens, input vector in the prediction phase
df = pd.read_csv(glob.glob("/home/lorenzo/arb_bwe/data_testing/sensor_fusion/prova_con_xsense/with_dv/xsense/*.csv")[0])
df = pd.DataFrame(df)
df = df[["ts", "vx", "vy", "vz"]]
data = df.to_numpy()

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
    cutoff = 2
    order = 10

    return butter_lowpass_filter(x, cutoff, fs, order, nyq)

out = np.zeros((data.shape[0], 4)); out[:,0] = data[:,0]
for i in range(1,data.shape[1]):
    out[:,i] = filtering(data[:,i])


fig, axes = plt.subplots(3,2)
axes[0,0].plot(df["ts"], df["vx"]); axes[0,1].plot(out[:,0], out[:,1])
axes[1,0].plot(df["ts"], df["vy"]); axes[1,1].plot(out[:,0], out[:,2])
axes[2,0].plot(df["ts"], df["vz"]); axes[2,1].plot(out[:,0], out[:,3])

#integration
P = np.zeros(out[:,1:].shape); 
ts = 1./xsense_freq
for i in range(len(out[:,0])):
    P[i,:] = P[i-1,:] + out[i,1:] * ts

fig, axes = plt.subplots(3,1)
axes[0].plot(df["ts"], P[:,0])
axes[1].plot(df["ts"], P[:,1])
axes[2].plot(df["ts"], P[:,2])

plt.show()
