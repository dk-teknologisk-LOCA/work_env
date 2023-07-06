from random import randrange
import pandas as pd
import pathlib
from filterpy.kalman import ExtendedKalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints , JulierSigmaPoints
import matplotlib.pyplot as plt
from numpy import array, arange, asarray, sqrt, eye, diag, linalg, ones, transpose, subtract, pi, sin, matmul, vstack, hstack, loadtxt, mean, linspace
from numpy.random import randn, normal, uniform

####HERE THE UKF IS COUPLED WITH ACCCELERATION AVERAGING TAKING DATA BOTH FROM THE PAST AND FUTURE AND IT DOENST IMPROVE IT 

def read_data_creaform():
    filepath = pathlib.Path(__file__).parent / "data/creaform/square_run_2.CSV"
    df = pd.read_csv(filepath, skiprows=1, usecols=[0,1,2,3,7], header=0, names=['TS', 'TX', 'TY', 'TZ', 'Valid'])
    full_pos = pd.DataFrame(df).to_numpy(); full_pos[:,1:-1] = full_pos[:,1:-1]/1000.
    t = full_pos[:,0]; full_pos = full_pos[:,1:]
    he_calib_mat = loadtxt(pathlib.Path(__file__).parent.parent / "hand_eye_calibration/he_calibration_matrix.txt")
    rot_pos = matmul(he_calib_mat, hstack((full_pos[:,:-1], ones(full_pos[:,0].shape).reshape(-1,1))).T).T
    full_pos = hstack((rot_pos[:,:-1], full_pos[:,-1].reshape(-1,1)))
    #pos = np.vstack((full_pos[:,0], full_pos[:,-1])).T

    return full_pos, t

def read_true_data():
    #TX, TY, TZ for the robot
    filename = pathlib.Path(__file__).parent / "data/ROS_files/run_2/run_2_tf.csv"
    data = pd.DataFrame(pd.read_csv(filename))
    data = data[["parent_frame", "child_frame", "posx", "posy", "posz"]]
    data = data[["posx", "posy", "posz"]].to_numpy()

    return data

def plotit(true_data, creaform_data, recon_pos, recon_acc):
    dims = ["x", "y", "z"]
    for i in range(len(recon_pos[0,:])):
        fig, ax = plt.subplots(2)
        fig.suptitle(dims[i])
        ax[0].plot(recon_pos[:,i], label = "recon pos")
        ax[0].plot(creaform_data[:,i], '+', label = "crea pos")
        ax[0].grid()
        ax0_bis = ax[0].twiny()
        ax0_bis.plot(true_data[:,i], 'r--', label = "UR pos")
        ax[1].plot(recon_acc[:,i], label = "recon acc")
        ax[1].grid()
        #ax[2].plot(true_data-recon_data)
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
        self.j = 0
        k = 1
        self.covs = [self.UKF.P]
        for self.i in range(1, len(t)):
            z = self.pos[self.i,:] 
            self.UKF.predict()
            xs_arr = asarray(self.xs)
            self.UKF.x[2] = mean(xs_arr[self.i-1-self.j:self.i, 2], 0)
            #if z[1] != 0:
            #    self.UKF.update(z[0]) 
            if len(t) - self.i < k:
                    print("QUI")
                    k = k-1  
            if z[1] != 0:
                points = []
                if self.j == 0:
                    points.append(self.pos[self.i, :-1])
                else:
                    for q in range(self.j):
                        if self.pos[self.i-q, -1] != 0:
                            points.append(self.pos[self.i-q, :-1])
                for p in range(k):
                    if self.pos[self.i+p, -1] != 0:
                        points.append(self.pos[self.i+p, :-1])
                #print(len(t))
                #print(self.i)
                points.reverse()
                z = mean(points)
                print(z)
                self.UKF.update(z) 
                if self.j<10:
                    self.j += 1  



            self.xs.append(self.UKF.x)
            self.covs.append(self.UKF.P)
            #if self.j < 100:
            #    self.j += 1
        
        self.xs, P, K = self.UKF.rts_smoother(asarray(self.xs), asarray(self.covs))
        return self.xs

    def create_UKF(self):
        self.sigmas = MerweScaledSigmaPoints(self.m, alpha=.1, beta=2., kappa=1.)
        self.UKF = UnscentedKalmanFilter(dim_x=self.m, dim_z=self.n, dt = self.dt, hx=self.hx, fx=self.fx, points=self.sigmas)
        self.UKF.x = array([0.1, 0.1, 0.1])
        self.UKF.P *= 50
        z_std = 0.001**2.; self.UKF.R = diag([z_std])
        state_std = 1**2.; self.UKF.Q = diag([state_std, state_std, state_std]) #UKF.Q = Q_discrete_white_noise(2, dt=dt, var=state_std)

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
rec_x = Ukf.run_UKF_cycle(posx); rec_y = Ukf.run_UKF_cycle(posy); rec_z = Ukf.run_UKF_cycle(posz)
recon_pos = hstack((rec_x[:,0].reshape(-1,1), rec_y[:,0].reshape(-1,1), rec_z[:,0].reshape(-1,1)))
recon_acc = hstack((rec_x[:,2].reshape(-1,1), rec_y[:,2].reshape(-1,1), rec_z[:,2].reshape(-1,1)))

plotit(true_data, pos, recon_pos, recon_acc)

fig, axes = plt.subplots(3)
axes[0].plot(rec_x[:,2]), axes[1].plot(rec_y[:,2]), axes[2].plot(rec_z[:,2])

plt.show()
