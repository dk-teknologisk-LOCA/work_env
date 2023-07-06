from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sg

#Data load and preparation
def butter_lowpass_filter(data, cutoff, fs, order, nyq):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = sg.butter(order, normal_cutoff, btype='low', analog=False)
    y = sg.filtfilt(b, a, data)
    return y

def filtering(x):
    xsense_freq = 80.
    # Filter requirements.
    #T = 5.0         # Sample Period
    fs = xsense_freq       # sample rate, Hz
    nyq = 0.5 * fs + 2      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hznyq = 0.5 * fs  # Nyquist Frequencyorder = 2       # sin wave can be approx represented as quadratic
    n = len(x) # total number of samples
    cutoff = 2
    order = 2

    return butter_lowpass_filter(x, cutoff, fs, order, nyq)

def load_xsens(filename):
    #ts, fax, fay, faz for xsens, input vector in the prediction phase
    df = pd.read_csv(filename)
    data = pd.DataFrame(df)

    return data[["ts", "fax", "fay", "faz"]]

data = load_xsens("/home/lorenzo/arb_bwe/xsense_creaform_sf/combined_data/ROS_files/test_1/test_1_free_acceleration.csv")

datax = data[["ts", "fax"]]

out = filtering(datax.to_numpy()[:,-1])

dt = 1./80.

velx = zeros(out.shape)
posx = zeros(out.shape)
print(velx.shape)
for i in range(1, len(out)):
    velx[i] = velx[i-1] + out[i] * dt
    posx[i] = posx[i-1] + velx[i] * dt + 0.5 * out[i] * dt ** 2

linevelm = (velx[-1] - velx[0])/(datax["ts"].to_numpy()[-1] - datax["ts"].to_numpy()[0])
lineveltot = linevelm * datax["ts"].to_numpy()
velx = velx - lineveltot

for i in range(1, len(out)):
    posx[i] = posx[i-1] + velx[i] * dt + 0.5 * out[i] * dt ** 2

lineposm = (posx[-1] - posx[0])/(datax["ts"].to_numpy()[-1] - datax["ts"].to_numpy()[0])
linepostot = lineposm * datax["ts"].to_numpy()
posx = posx - linepostot


fig, axes = plt.subplots(3,1)
axes[0].plot(datax["ts"], out)
axes[1].plot(datax["ts"], velx)
axes[2].plot(datax["ts"], posx)
plt.show()