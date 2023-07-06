import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

global he_calibmat
he_calibmat = np.loadtxt("robotics/hand_eye_calibration/he_calibration_matrix.txt")

def load_robot(filename):
    #TX, TY, TZ, Rx, Ry, Rz, Rw for the robot
    df = pd.read_csv(filename)
    data = pd.DataFrame(df)
    data = data[["parent_frame", "child_frame", "posx", "posy", "posz", "rotx", "roty", "rotz", "rotw"]]
    data.drop(data[data.parent_frame != 'base'].index, inplace=True)
    data = data[["posx", "posy", "posz", "rotx", "roty", "rotz", "rotw"]]
    data = data.to_numpy()
    #print("data \n", data)
    return data

def load_creaform(filename):
    global he_calibmat
    df = pd.read_csv(filename, skiprows=1, usecols=[0,1,2,3,4,5,6,7], header=0, names=['TS', 'TX', 'TY', 'TZ', 'RX', 'RY', 'RZ', 'Valid'])
    data = pd.DataFrame(df)
    #TSC, TX, TY, TZ, RX, RY, RZ, Valid
    data = data.to_numpy()
    #read the HE calibration matrix data
    H_W_C = he_calibmat
    
    data[:,1:4] = np.matmul(H_W_C, np.hstack([data[:,1:4], np.ones(data[:,0].shape).reshape(len(data[:,0]), 1)]).T).T[:,:-1]
    data[:,:-1] = data[:,:-1]/1000.
    return data

data_robot = load_robot("")