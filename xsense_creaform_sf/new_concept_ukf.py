from numpy import *
import pandas as pd
import pathlib
from filterpy.kalman import *
from filterpy.common import *
import matplotlib.pyplot as plt

def read_crea():
    #creaform data n x 5: ts x y z valid
    filepath = pathlib.Path(__file__).parent.parent / "combined_data/creaform/run_1.CSV"
    df = pd.read_csv(filepath, skiprows=1, usecols=[0,1,2,3,7], header=0, names=['TS', 'TX', 'TY', 'TZ', 'Valid'])
    full_pos = pd.DataFrame(df).to_numpy(); full_pos[:,1:-1] = full_pos[:,1:-1]/1000.
    t = full_pos[:,0]; full_pos = full_pos[:,1:]
    he_calib_mat = loadtxt(pathlib.Path(__file__).parent.parent.parent / "hand_eye_calibration/he_calibration_matrix.txt")
    rot_pos = matmul(he_calib_mat, hstack((full_pos[:,:-1], ones(full_pos[:,0].shape).reshape(-1,1))).T).T
    full_pos = hstack((rot_pos[:,:-1], full_pos[:,-1].reshape(-1,1)))

    return hstack((t.reshape(-1,1), full_pos))

def read_xsens():
    #xsens data n x 4: ts fax fay faz 
    filepath = pathlib.Path(__file__).parent / "combined_data/xsens/run_1.CSV"
    data = pd.DataFrame(pd.read_csv(filepath))
    data = data[["ts", "fax", "fay", "faz"]].to_numpy()

    return data

def read_true_data():
    #TX, TY, TZ for the robot
    filename = pathlib.Path(__file__).parent / "data/ROS_files/run_2/run_2_tf.csv"
    data = pd.DataFrame(pd.read_csv(filename))
    data = data[["parent_frame", "child_frame", "posx", "posy", "posz"]]
    data = data[["posx", "posy", "posz"]].to_numpy()

    return data

def unite_measures():
    #read the data
    crea_data = read_crea(); xsens_data = read_xsens()
    print([crea_data[:,0], xsens_data[:,0]])

    #syncronise the data?
    
    #Unify measurement lenghts
    data_shape = crea_data.shape
    xsens_data = xsens_data[:data_shape[0], :]
    
    #prepare complete data
    stitched_data = zeros(data_shape[0], 8) #ts x y z valid xx yx zx
    stitched_data[:,:5] = crea_data
    #Get indexes of non valid creaform readings
    idxs = where(crea_data[:,-1] == 0)

    for i, val in enumerate(idxs):
        pre_point = crea_data[i-1, 1:4]
        stitched_data[i, 5:] = myUKF.model(pre_point, dt = 1/80)

    savetxt("stitched_data", stitched_data)

    datax = stitched_data[:, [0, 1, 4, 5]]
    datay = stitched_data[:, [0, 2, 4, 6]]
    dataz = stitched_data[:, [0, 3, 4, 7]]

    return datax, datay, dataz

def plotit(true_data, creaform_data, recon_pos):
    dims = ["x", "y", "z"]
    for i in range(len(recon_pos[0,:])):
        fig, ax = plt.subplots()
        fig.suptitle(dims[i])
        ax[0].plot(recon_pos[:,i], label = "recon pos")
        ax[0].plot(creaform_data[:,i], '+', label = "crea pos")
        ax[0].grid()
        ax0_bis = ax[0].twiny()
        ax0_bis.plot(true_data[:,i], 'r--', label = "UR pos")
        #ax[2].plot(true_data-recon_data)
        plt.legend(loc=2)

class myUKF:
    def __init__(self):
        self.dt = 1./80.
        self.m = 3 #num states
        self.n = 1 #num measures
        self.create_UKF()

    def create_UKF(self):
        self.sigmas = MerweScaledSigmaPoints(self.m, alpha=.1, beta=2., kappa=1.)
        self.UKF = UnscentedKalmanFilter(dim_x=self.m, dim_z=self.n, dt = self.dt, hx=self.hx, fx=self.fx, points=self.sigmas)
        self.UKF.x = array([0.1, 0.1, 0.1])
        self.UKF.P *= 50
        z_std = 0.01**2.; self.UKF.R = diag([z_std])
        state_std = 0.001**2.; self.UKF.Q = diag([state_std, state_std, state_std]) #UKF.Q = Q_discrete_white_noise(2, dt=dt, var=state_std)

    def run_UKF(self, input_data):
        self.data = input_data
        self.time = self.data[:,0]
        xs = []
        self.UKF.predict()
        for i, val in enumerate(self.time): #ts x y z valid xx yx zx
            if self.data[i, 4] == 1:
                self.UKF.update(self.data[i, 1])
            elif self.data[i, 4] == 0:
                self.UKF.update(self.data[i, 5])
            xs.append(self.UKF.x)
            self.UKF.predict()

        return asarray(xs)
    
    @staticmethod
    def model(x, dt):
        F = array([[1, dt, 0.5*dt**2], [0, 1, dt], [0, 0, 1]], dtype = float) #constant acceleration Newton model
        return F @ x

    def measure(self, x):
        return x

true_data = read_true_data()
datax, datay, dataz = unite_measures()

ukf = myUKF()
x_rec = ukf.run_UKF(datax); y_rec = ukf.run_UKF(datay); z_rec = ukf.run_UKF(dataz) #these have this structure: x, vx, ax
recon_pos = hstack((x_rec[:,0].reshape(-1,1), y_rec[:,0].reshape(-1,1), z_rec[:,0].reshape(-1,1)))

plotit(true_data, hstack((datax[:,2].reshape(-1,1), y_rec[:,2].reshape(-1,1), z_rec[:,2].reshape(-1,1))), recon_pos)