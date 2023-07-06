from random import randrange
import pandas as pd
import pathlib
from filterpy.kalman import ExtendedKalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints , JulierSigmaPoints
import matplotlib.pyplot as plt
from numpy import array, arange, asarray, sqrt, eye, diag, linalg, ones, transpose, subtract, pi, sin, matmul, vstack, hstack, loadtxt
from numpy.random import randn, normal, uniform
from rotate_xsens import *
import math

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

def read_true_data():
    #TX, TY, TZ for the robot
    filename = pathlib.Path(__file__).parent.parent / "data/ROS_files/run_1_tf/run_1_tf.csv"
    data = pd.DataFrame(pd.read_csv(filename))
    data = data[["parent_frame", "child_frame", "posx", "posy", "posz"]]
    data = data[["posx", "posy", "posz"]].to_numpy()

    return data

def plotit(true_data, creaform_data, recon_data, plot_true = True):
    dims = ["x", "y", "z"]
    for i in range(len(recon_data[0,:])):
        fig, ax = plt.subplots(1)
        fig.suptitle(dims[i]+"standard")
        ax.plot(recon_data[:,i], "r-", label="recon")
        ax.plot(creaform_data[:,i], 'g*', label = "true")
        ax.grid()
        ax.legend(loc = 4)
        if plot_true:
            ax0_bis = ax[0].twiny()
            ax0_bis.plot(true_data[:,i], 'b--', label = "UR pos")
            plt.legend()
        plt.legend(loc=2)
        
class myUKF:
    def __init__(self):
        self.dt = 1./80.
        self.m = 3 #num states
        self.n = 1 #num measures
        self.create_UKF()

    def run_UKF_cycle(self, pos_data):
        self.pos = pos_data
        self.xs = [self.UKF.x]
        self.covs = [self.UKF.P]
        for i in range(1, self.pos.shape[0]):
            z = self.pos[i,:] 
            self.UKF.predict()
            if z[1] != 0:
                self.UKF.update(z[0])   
            self.xs.append(self.UKF.x)
            self.covs.append(self.UKF.P)
            
        self.xs, P, K = self.UKF.rts_smoother(asarray(self.xs), asarray(self.covs))
        return asarray(self.xs)[:,0]

    def create_UKF(self):
        self.sigmas = MerweScaledSigmaPoints(self.m, alpha=0.1, beta=2.0, kappa=3.-self.m)
        self.UKF = UnscentedKalmanFilter(dim_x=self.m, dim_z=self.n, dt = self.dt, hx=self.hx, fx=self.fx, points=self.sigmas)
        self.UKF.x = array([0.1, 0.1, 0.1])
        self.UKF.P *= 50
        z_std = 0.01**2.; self.UKF.R = diag([z_std])
        state_std = 0.01**2.; self.UKF.Q = diag([state_std, state_std, state_std]) #UKF.Q = Q_discrete_white_noise(2, dt=dt, var=state_std)

    def hx(self, x):
        return x
    
    def fx(self, x, dt):
        F = array([[1, dt, 0.5*dt**2],[0, 1, dt],[0, 0, 1]], dtype = float)
        return F @ x

#data
pos, t = read_data_creaform()
posx = vstack((pos[:,0], pos[:,-1])).T
posy = vstack((pos[:,1], pos[:,-1])).T
posz = vstack((pos[:,2], pos[:,-1])).T

#True data
true_data = read_true_data()
#true_data = resample(true_data, 718)

#KF
Ukf = myUKF()
rec_x = Ukf.run_UKF_cycle(posx).reshape(-1,1)
rec_y = Ukf.run_UKF_cycle(posy).reshape(-1,1)
rec_z = Ukf.run_UKF_cycle(posz).reshape(-1,1)
recon_data = hstack((rec_x, rec_y, rec_z))

plotit(true_data, pos, recon_data, plot_true = False)

def load_xsens(filename):
    #ts, fax, fay, faz for xsens, input vector in the prediction phase
    df = pd.read_csv(filename)
    data = pd.DataFrame(df)

    return data[["timestamp","qw","qx","qy","qz"]]

def euler_from_quaternion(quat):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        w = quat[0]; x = quat[1]; y = quat[2]; z = quat[3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

#now for the orientation
rotated_xsens_orientation_data = rotate_xsens_orientation(load_xsens("/home/lorenzo/arb_bwe/creaform_sf/data/ROS_files/run_quat_1/run_quat_1_quaternion.csv"))
xsens_in_RPY = pd.DataFrame(index=range(rotated_xsens_orientation_data.shape[0]), columns=["timestamp","R","P","Y"])
for i in range(rotated_xsens_orientation_data.shape[0]):
    xsens_in_RPY.iloc[i][1:] = euler_from_quaternion(rotated_xsens_orientation_data.iloc[i][1:])
xsens_in_RPY["timestamp"] = rotated_xsens_orientation_data["timestamp"]
fig, axes = plt.subplots(3)
axes[0].plot(xsens_in_RPY["R"])
axes[1].plot(xsens_in_RPY["P"])
axes[2].plot(xsens_in_RPY["Y"])

plt.show()

