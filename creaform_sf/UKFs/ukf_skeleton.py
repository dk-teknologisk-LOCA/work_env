from numpy import array, arange, asarray, sqrt, eye, diag, linalg, ones, transpose, subtract, pi, sin, cos, exp
from numpy import loadtxt, vstack, hstack, matmul
from numpy.random import randn, normal, uniform
import math
import matplotlib.pyplot as plt
from filterpy.kalman import ExtendedKalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints 
from filterpy.common import Q_discrete_white_noise
from random import randrange
import pathlib
import pandas as pd

def read_data_creaform():
    #filepath = pathlib.Path(__file__).parent.parent / "data/creaform/center_on_targets_1.CSV"
    filepath = pathlib.Path(__file__).parent.parent.parent/"ringsted_data/ringsted_25_05_23/center_on_targets_2_bad_signal.CSV"
    df = pd.read_csv(filepath, skiprows=1, usecols=[0,1,2,3,7], header=0, names=['TS', 'TX', 'TY', 'TZ', 'Valid'])
    full_pos = pd.DataFrame(df).to_numpy(); full_pos[:,1:-1] = full_pos[:,1:-1]/1000.
    t = full_pos[:,0]; full_pos = full_pos[:,1:]
    he_calib_mat = loadtxt(pathlib.Path(__file__).parent.parent.parent / "hand_eye_calibration/he_calibration_matrix.txt")
    rot_pos = matmul(he_calib_mat, hstack((full_pos[:,:-1], ones(full_pos[:,0].shape).reshape(-1,1))).T).T
    full_pos = hstack((rot_pos[:,:-1], full_pos[:,-1].reshape(-1,1)))

    return full_pos, t

def hx(x):
    return x

def fx(x, dt):
    F = array([[1, dt, 0.5 * dt ** 2],[0, 1, dt], [0, 0, 1]], dtype = float)
    return F @ x

    #return array(x + (x-u)*dt, dtype = float)

#data
dt = 1/80
t = arange(0, 10, dt)
pos = ones((len(t), 2))

#Some various signals
#pos[:,0] = 1./20.*t**2 + t + 10 + normal(0,0.1,len(t))
#pos[:,0] = 1./10.*cos(t)**2 + 10 + normal(0,0.1,len(t))
pos[:,0] = 1./10.*exp(t/10)**2 + 10 + normal(0,0.001,len(t))
#pos[:,0] = normal(0,0.01,len(t))
#pos[:,0] = sin(t) + normal(0,0.01,len(t))

#Make some points unseeable by setting them to 0
randvec = asarray([randrange(0, len(t)) for i in range(100)])
pos[randvec, 1] = 0

pos_full, t = read_data_creaform()
pos = hstack((pos_full[:, 0].reshape(-1, 1),  pos_full[:, -1].reshape(-1, 1)))

#KF
sigmas = MerweScaledSigmaPoints(3, alpha=.1, beta=2., kappa=1.)
UKF = UnscentedKalmanFilter(dim_x=3, dim_z=1, dt = dt, hx=hx, fx=fx, points=sigmas)
UKF.x = array([0.1, 0.1, 0.1])
UKF.P *= 50
z_std = 0.001**2.; UKF.R = diag([z_std])
state_std = 0.1**2.; UKF.Q = diag([state_std, state_std, state_std]) #UKF.Q = Q_discrete_white_noise(2, dt=dt, var=state_std)

xs, track, covdet = [], [], []
for i in range(len(t)):
    z = pos[i,:]
    track.append(z[0])
    UKF.predict()
    if z[1] != 0:
        UKF.update([z[0]])   
    xs.append(UKF.x)
    

xs = asarray(xs)
track = asarray(track)
cov = asarray(covdet)
fig = plt.figure()
plt.plot(xs[:,0], label="recon")
plt.plot(pos[:,0], '.', markersize=4, label = "true")
plt.grid()
plt.legend(loc=2)

fig = plt.figure()
plt.plot(xs[:,1])
fig = plt.figure()
plt.plot(xs[:,2])
plt.show()
