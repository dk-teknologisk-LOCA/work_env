from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R 

# Here  we try to reuse the Xsens in a more complete way each datapoint must be rotated according to the CORRESPONDING orientation data back to the ENU frame. 
# Then position according to ENU is calculated.
#thhe orientation data  fgivenn by the xsens is already the anngle necessary too go from the xsens back to eenu. Aka, R_xb_ENU.
#So, the materix trasformtion should be: R_xacc_ENU = R_xacc_xb * R_xb_ENU. R_xacc_xb is the reading of the acceleration (in this case we will use the free acceleration. 
# We will try also with the normal acceleration).

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

#calculate trajectory
def calculate_trajectory(rotated_data):
    dt = 1./16.
    rotated_data["vx"] = 0; rotated_data["vy"] = 0; rotated_data["vz"] = 0
    rotated_data["px"] = 0; rotated_data["py"] = 0; rotated_data["pz"] = 0 
    data = rotated_data.to_numpy()
    #       0       1   2   3      4    5     6     7     8    9    10    11    12    13
    #"timestamp","qw","qx","qy","qz", "ax", "ay", "az", "vx", "vy", "vz", "px", "py", "pz"
    for i in range(1, data.shape[0]):
        data[i, 8:11] = data[i-1, 8:11] + data[i, 5:8] * dt
        #data[i, 9] = data[i-1, 9] + data[i, 6] * dt
        #data[i, 10] = data[i-1, 10] + data[i, 7] * dt
        #data[i, 11] = data[i-1, 11] + data[i, 8] * dt

        data[i, 11:]  = data[i-1, 11:] + data[i, 8:11] * dt + 0.5 * data[i, 5:8] * dt ** 2.
        #data[i, 12] = data[i-1, 12] + data[i, 9] * dt + 0.5 * data[i, 6] * dt ** 2.
        #data[i, 13] = data[i-1, 13] + data[i, 10] * dt + 0.5 * data[i, 7] * dt ** 2.
        #data[i, 14] = data[i-1, 14] + data[i, 11] * dt + 0.5 * data[i, 8] * dt ** 2.

    trajectory = pd.DataFrame(data[:, [0, 11, 12, 13]], columns=["time_ref", "px", "py", "pz"])
    return trajectory

def plotdata(data1, data2, data3, title):
    fig, axes = plt.subplots(3)
    axes[0].plot(data1["fax"])
    axes[0].plot(data2["fax"])
    axes[0].plot(data3["fax"])
    axes[0].legend(["OG", "Filtered", "Rotated"])

    axes[1].plot(data1["fay"])
    axes[1].plot(data2["fay"])
    axes[1].plot(data3["fay"])
    axes[1].legend(["OG", "Filtered", "Rotated"])
    
    axes[2].plot(data1["faz"])
    axes[2].plot(data2["faz"])
    axes[2].plot(data3["faz"])
    axes[2].legend(["OG", "Filtered", "Rotated"])

    return fig 

#plot trajectory
def plottraj(data, title):
    figure, axes = plt.subplots(3)
    figure.suptitle(title)
    axes[0].plot(data.to_numpy()[:,0])
    axes[1].plot(data.to_numpy()[:,1])
    axes[2].plot(data.to_numpy()[:,2])

    pos = data.to_numpy()
    fig3 = plt.figure()
    axes3 = fig3.add_subplot(projection='3d')
    axes3.plot(pos[:,0]*1000, pos[:,1]*1000, pos[:,2]*1000)
    axes3.scatter3D(pos[0,0]*1000, pos[0,1]*1000, pos[0,2]*1000, '.', label="Init Point")
    axes3.scatter3D(pos[-1,0]*1000, pos[-1,1]*1000, pos[-1,2]*1000, '+', label="Final Point")
    axes3.legend()
    axes3.set_xlabel('X [mm]')
    axes3.set_ylabel('Y [mm]')
    axes3.set_zlabel('Z [mm]')

    return figure

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
        if tempnorm > 100: #for human movements. his includes also quasi sharp turns and movement inversions
        #if abs(tempnorm - meanval)>1:
            #print("STO QUI ", i)
            tempdata[i, 5:] = meanval
    #plt.plot(tempdata[:,5])
    #return smoothen(pd.DataFrame(tempdata, columns=data.columns))
    return pd.DataFrame(tempdata, columns=data.columns)

def pos_correction(data):
    tempdata = data.to_numpy()
    run_time = (tempdata[-1, 0] - tempdata[0, 0]) * 10 ** (-9)
    for i in range(1, len(tempdata[0, :])):
        incl = (tempdata[-1, i] - tempdata[0, i])/run_time
        line = incl * (tempdata[:, 0] - tempdata[0, 0]) * 10 ** (-9)
        tempdata[:, i] = tempdata[:, i] - line

    return pd.DataFrame(tempdata, columns=["time_ref", "px", "py", "pz"])

if __name__ == "__main__":
    filename = "/home/lorenzo/arb_bwe/xsens/data/ROS_files/run_full_1/run_full_1_full.csv"
    data = load_xsens(filename)
    #print("OG data\n", data)
    #Filter acceleration
    filtered_data = filterit(data)
    #rotate data
    rot_data = rotate_points(filtered_data)
    #print(filtered_data)
    #print("rotated data\n", rotated_data)
    trajectory = calculate_trajectory(rot_data)
    #trajectory = pos_correction(trajectory) #uncomment to force initial position = ending position
    #print(corr_traj)
    plotdata(data[["fax", "fay", "faz"]], filtered_data[["fax", "fay", "faz"]], rot_data[["fax", "fay", "faz"]], "Data Plot")
    plottraj(trajectory[["px", "py", "pz"]], "Trajectory")
    plt.show()


