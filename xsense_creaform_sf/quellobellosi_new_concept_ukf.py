from numpy import array, arange, asarray, sqrt, eye, diag, linalg, ones, transpose, subtract, pi, sin, cos, exp
from numpy import loadtxt, vstack, hstack, matmul, copy
from numpy.random import randn, normal, uniform
import math
import matplotlib.pyplot as plt
from filterpy.kalman import ExtendedKalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints 
from filterpy.common import Q_discrete_white_noise
from random import randrange
import pathlib
import pandas as pd
import scipy.signal as sg


#Ths is a result ofa  revelation that hsould have been obvious
# here the state vector is [pos, vel, acc] and the measuremennt vectos is a 
# joined vector of [creaform position, xsense acceleration, creaform valid flag]
# The filter for unvalid data poits from the creafom is moved to the 
# H function. 
# In theory this system should be CORRECT, while he previous ones where injecting 
# measurements in the wrong state. 
# As of now it  is only for x coordinate but can be expanded. 
# NOTE that the problem off creating a common Point Of View 
# for the XSens and the Creaform stil exists.

#Data load and preparation
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
    cutoff = 3
    order = 10

    return butter_lowpass_filter(x, cutoff, fs, order, nyq)

def read_data_creaform():
    #filepath = pathlib.Path(__file__).parent.parent / "data/creaform/center_on_targets_1.CSV"
    filepath = pathlib.Path(__file__).parent.parent.parent/"ringsted_data/ringsted_25_05_23/center_on_targets_1.CSV"
    df = pd.read_csv(filepath, skiprows=1, usecols=[0,1,2,3,7], header=0, names=['TS', 'TX', 'TY', 'TZ', 'Valid'])
    full_pos = pd.DataFrame(df).to_numpy(); full_pos[:,1:-1] = full_pos[:,1:-1]/1000.
    t = full_pos[:,0]; full_pos = full_pos[:,1:]
    he_calib_mat = loadtxt(pathlib.Path(__file__).parent.parent.parent / "hand_eye_calibration/he_calibration_matrix.txt")
    rot_pos = matmul(he_calib_mat, hstack((full_pos[:,:-1], ones(full_pos[:,0].shape).reshape(-1,1))).T).T
    full_pos = hstack((rot_pos[:,:-1], full_pos[:,-1].reshape(-1,1)))

    return full_pos, t

def load_xsens(filename):
    #ts, fax, fay, faz for xsens, input vector in the prediction phase
    df = pd.read_csv(filename)
    data = pd.DataFrame(df)
    data = data[["ts", "fax", "fay", "faz"]]
    data = data.to_numpy()

    out = copy(data)
    for i in range(1,data.shape[1]):
        out[:,i] = filtering(data[:,i])

    return data

def hx(x):
    if x[-1] == 0:
        H = array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
    else:
        H = array([[1, 0, 0], [0, 0, 0], [0, 0, 1]])
    return H @ x #get position ad acceleration

def fx(x, dt):
    F = array([[1, dt, 0.5 * dt ** 2],[0, 1, dt], [0, 0, 1]], dtype = float)
    return F @ x

    #return array(x + (x-u)*dt, dtype = float)

#data
dt = 1/80

pos_full, t = read_data_creaform()
acc_full = load_xsens()
#pos = hstack((pos_full[:, 0].reshape(-1, 1),  pos_full[:, -1].reshape(-1, 1)))
measvec = hstack((pos_full[:, 0].reshape(-1, 1), acc_full[:,1].reshape(-1,1), pos_full[:, -1].reshape(-1, 1)))

#KF
sigmas = MerweScaledSigmaPoints(3, alpha=.1, beta=2., kappa=1.)
UKF = UnscentedKalmanFilter(dim_x=3, dim_z=2, dt = dt, hx=hx, fx=fx, points=sigmas)
UKF.x = array([0.1, 0.1, 0.1])
UKF.P *= 50
z_std = 0.001**2.; UKF.R = diag([z_std, z_std])
state_std = 0.01**2.; UKF.Q = diag([state_std, state_std, state_std]) #UKF.Q = Q_discrete_white_noise(2, dt=dt, var=state_std)

xs, track, covdet = [], [], []
for i in range(len(t)):
    z = measvec[i,:] 
    track.append(z[0])
    UKF.predict()
    UKF.update(z)   
    xs.append(UKF.x)

xs = asarray(xs)
track = asarray(track)
cov = asarray(covdet)
fig = plt.figure()
plt.plot(xs[:,0], label="recon")
plt.plot(measvec[:,0], '.', markersize=4, label = "true")
plt.grid()
plt.legend(loc=2)
plt.show()
