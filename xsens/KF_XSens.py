from random import randrange
import pandas as pd
import pathlib
from filterpy.kalman import ExtendedKalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints , JulierSigmaPoints
import matplotlib.pyplot as plt
from numpy import *
from numpy.random import randn, normal, uniform
import math

#Utils functions for operations done with the quaternions
def quaternion_multiply(Q0,Q1):
    """
    Multiplies two quaternions.
 
    Input
    :param Q0: A 4 element array containing the first quaternion (q01,q11,q21,q31) 
    :param Q1: A 4 element array containing the second quaternion (q02,q12,q22,q32) 
 
    Output
    :return: A 4 element array containing the final quaternion (q03,q13,q23,q33) 
 
    """
    # Extract the values from Q0
    w0 = Q0[0]
    x0 = Q0[1]
    y0 = Q0[2]
    z0 = Q0[3]
     
    # Extract the values from Q1
    w1 = Q1[0]
    x1 = Q1[1]
    y1 = Q1[2]
    z1 = Q1[3]
     
    # Computer the product of the two quaternions, term by term
    Q0Q1_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    Q0Q1_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    Q0Q1_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    Q0Q1_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
     
    # Create a 4 element array containing the final quaternion
    final_quaternion = array([Q0Q1_w, Q0Q1_x, Q0Q1_y, Q0Q1_z])
     
    # Return a 4 element array containing the final quaternion (q02,q12,q22,q32) 
    return final_quaternion

def invert_quat(Q):
    Q_out = Q
    Q_out[1] = -Q[1]
    Q_out[2] = -Q[2]
    Q_out[3] = -Q[3]
    return Q_out

def get_quaternion_from_euler(RPY):
  """
  Convert an Euler angle to a quaternion.
   
  Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.
 
  Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
  """
  roll = RPY[0]; pitch = RPY[1]; yaw = RPY[2]
  qx = sin(roll/2) * cos(pitch/2) * cos(yaw/2) - cos(roll/2) * sin(pitch/2) * sin(yaw/2)
  qy = cos(roll/2) * sin(pitch/2) * cos(yaw/2) + sin(roll/2) * cos(pitch/2) * sin(yaw/2)
  qz = cos(roll/2) * cos(pitch/2) * sin(yaw/2) - sin(roll/2) * sin(pitch/2) * cos(yaw/2)
  qw = cos(roll/2) * cos(pitch/2) * cos(yaw/2) + sin(roll/2) * sin(pitch/2) * sin(yaw/2)
 
  return [qx, qy, qz, qw]

#read data IN:csv OUT: pandas with freeacc annd orientation
def load_xsens(filename):
    #return pd.DataFrame(pd.read_csv(filename))[["timestamp","qw","qx","qy","qz", "ax", "ay", "az"]]
    return pd.DataFrame(pd.read_csv(filename))[["time_ref","qw","qx","qy","qz", "fax", "fay", "faz"]].dropna(axis=0)

#rotate each freeacc datapoint to ENU
def rotate_points(data):
    rotated_data = copy(data)
    #rotate all data
    for i in range(data.shape[0]):
        quat_temp = data.iloc[i][1:5].tolist() # w, x, y, z.
        inv_quat_temp = invert_quat(quat_temp) # w, x, y, z.
        rotated_data[i, 5:] = quaternion_multiply(quaternion_multiply(inv_quat_temp, [0] + data.iloc[i][5:].tolist()), quat_temp)[1:]
    rotated_data = pd.DataFrame(rotated_data, columns = data.columns)
    
    return rotated_data

def smoothen(data):
    temp_data = data.to_numpy()
    kernel_size = 3
    kernel = ones(kernel_size) / kernel_size
    for i in range(temp_data[:,:5].shape[1], temp_data.shape[1]):
        temp_data[:,i] = convolve(temp_data[:,i], kernel, mode='same')
    
    return pd.DataFrame(temp_data, columns=data.columns)

def filterit(data):
    #remove outliers
    tempdata = data.to_numpy()
    meanval = mean(linalg.norm(tempdata[:, 5:], axis = 1), axis = 0)
    for i in range(1, len(tempdata)-1):
        tempnorm = linalg.norm(tempdata[i, 5:])
        if tempnorm > 3:
        #if abs(tempnorm - meanval)>1:
            #print("STO QUI ", i)
            tempdata[i, 5:] = meanval
    #plt.plot(tempdata[:,5])
    return smoothen(pd.DataFrame(tempdata, columns=data.columns))
    #return pd.DataFrame(tempdata, columns=data.columns)

def pos_correction(data, time_ref):
    run_time = (time_ref[-1] -time_ref[0]) * 10 ** (-9)
    for i in range(len(data[0, :])):
        incl = (data[-1, i] - data[0, i])/run_time
        print(incl)
        line = incl * (time_ref[:] -time_ref[0]) * 10 ** (-9)
        data[:, i] = data[:, i] - line
    
    return data

class myUKF:
    def __init__(self):
        self.dt = 1./16.
        self.A =  array([[1, 0, 0, self.dt, 0, 0, 0.5 * self.dt ** 2, 0, 0], 
                         [0, 1, 0, 0, self.dt, 0, 0, 0.5 * self.dt ** 2, 0], 
                         [0, 0, 1, 0, 0, self.dt, 0, 0, 0.5 * self.dt ** 2], 
                         [0, 0, 0, 1, 0, 0, self.dt, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, self.dt, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0, self.dt],
                         [0, 0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        self.H = array([[0,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,0,1]]).reshape(3,-1)
        self.Q = diag([10, 10, 10, 10, 10, 10, 0.01, 0.01, 0.01])
        self.R = diag([0.1, 0.1, 0.1])
        print("H mat \n", self.H.shape, "\nA mat \n", self.A.shape, "\nQ mat \n", self.Q.shape, "\nR mat \n", self.R.shape)

    def start_KF(self, data):
        self.x_est = ones((data.shape[0], 9)) * 0.01
        self.P = ones((9,9)) * 0.1
        for i in range(1, data.shape[0]):
            """Predict"""
            x_pre = matmul(self.A, self.x_est[i-1, :])
            print("x_pre", x_pre.shape)
            P_pre = matmul(matmul(self.A, self.P), self.A.T)+ self.Q
            print("P_pre", P_pre.shape)

            """Update"""
            K_g = matmul(matmul(P_pre, self.H.T), linalg.inv(matmul(matmul(self.H, P_pre), self.H.T) + self.R))
            print("K_g", K_g.shape)
            #print("KG", K_g)
            self.x_est[i,:] = x_pre + matmul(K_g, (data[i] - matmul(self.H, x_pre)))
            #print((eye(3,3) - matmul(self.H, K_g)))
            self.P = matmul((eye(9,9) - matmul(K_g, self.H)), P_pre)
        
        return self.x_est
    
if __name__ == "__main__":
    filename = "/home/lorenzo/arb_bwe/xsens/data/ROS_files/run_full_1/run_full_1_full.csv"
    data = load_xsens(filename)
    print("OG data\n", data)
    #Filter acceleration
    filtered_data = filterit(data)
    #rotate data
    rot_data = rotate_points(filtered_data)

    #KF
    Ukf = myUKF()
    rec_data = Ukf.start_KF(rot_data[["fax", "fay", "faz"]].to_numpy())
    
    pos = rec_data[:,:3]
    pos = pos_correction(pos, data["time_ref"].to_numpy()) #uncomment to force initial position = ending position
    
    vel = rec_data[:,3:6]
    acc =  rec_data[:,6:]

    #OG acc vs recon acc
    fig1, axes1 = plt.subplots(3)
    axes1[0].plot(rot_data["fax"])
    axes1[0].plot(acc[:,0], "--")
    axes1[1].plot(rot_data["fay"])
    axes1[1].plot(acc[:,0], "--")
    axes1[2].plot(rot_data["faz"])
    axes1[2].plot(acc[:,0], "--")
    fig1.suptitle("OG acc vs recon acc")


    fig2, axes2 = plt.subplots(3)
    axes2[0].plot(pos[:,0])
    axes2[1].plot(pos[:,1])
    axes2[2].plot(pos[:,2])
    fig2.suptitle("Positions")

    fig3 = plt.figure()
    axes3 = fig3.add_subplot(projection='3d')
    axes3.plot(pos[:,0]*1000, pos[:,1]*1000, pos[:,2]*1000)
    axes3.scatter3D(pos[0,0]*1000, pos[0,1]*1000, pos[0,2]*1000, '.', label="Init Point")
    axes3.scatter3D(pos[-1,0]*1000, pos[-1,1]*1000, pos[-1,2]*1000, '+', label="Final Point")
    axes3.legend()
    axes3.set_xlabel('X [mm]')
    axes3.set_ylabel('Y [mm]')
    axes3.set_zlabel('Z [mm]')
    plt.show()