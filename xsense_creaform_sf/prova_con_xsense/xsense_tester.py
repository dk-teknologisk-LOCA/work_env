import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

n = 9

def rpy_to_quat(r, p, y):
    r0 = np.cos(r * 0.5) * np.cos(p * 0.5) * np.cos(y * 0.5) + np.sin(r * 0.5) * np.sin(p * 0.5) * np.sin(y * 0.5)
    r1 = np.sin(r * 0.5) * np.cos(p * 0.5) * np.cos(y * 0.5) - np.cos(r * 0.5) * np.sin(p * 0.5) * np.sin(y * 0.5)
    r2 = np.cos(r * 0.5) * np.sin(p * 0.5) * np.cos(y * 0.5) + np.sin(r * 0.5) * np.cos(p * 0.5) * np.sin(y * 0.5)
    r3 = np.cos(r * 0.5) * np.cos(p * 0.5) * np.sin(y * 0.5) - np.sin(r * 0.5) * np.sin(p * 0.5) * np.cos(y * 0.5)

    quat_new = np.array([r0, r1, r2, r3])

    return quat_new

def quat_to_rpy(quaternione):
    r0 = quaternione[0]; r1 = quaternione[1]; r2 = quaternione[2]; r3 = quaternione[3]

    r_new = np.arctan(2 * (r0 * r1 + r2 * r3) / (r0 ** 2 - r1 ** 2 - r2 ** 2 + r3 ** 2))
    p_new = np.arcsin(2 * (r0 * r2 - r1 * r3))
    y_new = np.arctan(2 * (r0 * r3 + r1 * r2) / (r0 ** 2 + r1 ** 2 - r2 ** 2 - r3 ** 2))

    return r_new, p_new, y_new

def quatmul(q1, q2):
    r0 = q1[0]; r1 = q1[1]; r2 = q1[2]; r3 = q1[3]
    s0 = q2[0]; s1 = q2[1]; s2 = q2[2]; s3 = q2[3]

    t0 = r0*s0 - r1*s1 - r2*s2 - r3*s3
    t1 = r0*s1 + r1*s0 - r2*s3 + r3*s2
    t2 = r0*s2 + r1*s3 + r2*s0 - r3*s1
    t3 = r0*s3 - r1*s2 + r2*s1 + r3*s0

    new_quat = np.array([t0, t1, t2, t3])

    return new_quat

def quatconj(q):
    r0 = q[0]; r1 = q[1]; r2 = q[2]; r3 = q[3]
    newvec = np.array([r0, -r1, -r2, -r3])
    
    return newvec

def fx_fun(x, dt, u):
    #states vector: sx, sy, sz, r, p, y, vx, vy, vz
    #input vector: fax, fay, faz, wx, wy, wz
 
    xout= np.zeros(n)

    xout[0] = x[0] + x[6] * dt + 0.5 * u[0] * dt ** 2
    xout[1] = x[1] + x[7] * dt + 0.5 * u[1] * dt ** 2
    xout[2] = x[2] + x[8] * dt + 0.5 * u[2] * dt ** 2

    xout[6] = x[6] + u[0] * dt
    xout[7] = x[7] + u[1] * dt
    xout[8] = x[8] + u[2] * dt


    return xout

def load_xsens(filename):
    #AX, AY, AZ, WX, WY, WZ, Ox, Oy, Oz, Ow for xsens, input vector in the prediction phase

    df = pd.read_csv(filename)
    data = pd.DataFrame(df)
    data = data[["ts", "fax", "fay", "faz"]]
    data = data.to_numpy()

    #plot xsense data
    fig0, axes0 = plt.subplots(3,1)
    fig0.suptitle("Xsense acc pre filter")
    axes0[0].plot(data[:,1]); axes0[0].set_ylabel("x")
    axes0[1].plot(data[:,2]); axes0[1].set_ylabel("y")
    axes0[2].plot(data[:,3]); axes0[2].set_ylabel("z")

    acc_cut_off = 0.4 #tuning variable depending on the platform used and on the acceleration/speed of the platform and motion themselves
    acc_vec = np.sqrt(data[:,1]**2 + data[:,2]**2 + data[:,3]**2)
    med_acc = np.mean(acc_vec)
    acc_vec= acc_vec-med_acc
    for i in range(len(acc_vec)):
        if np.abs(acc_vec[i] - med_acc) < acc_cut_off:
            data[i,1:4] = np.zeros(data[i,1:4].shape)

    #plot xsense data
    fig0, axes0 = plt.subplots(3,1)
    fig0.suptitle("Xsense acc post filter")
    axes0[0].plot(data[:,1]); axes0[0].set_ylabel("x")
    axes0[1].plot(data[:,2]); axes0[1].set_ylabel("y")
    axes0[2].plot(data[:,3]); axes0[2].set_ylabel("z")

    return data

data = load_xsens("data_testing/sensor_fusion/prova_con_xsense/xsense/run_1_imu_acceleration.csv")
dt = 1./100.

x = np.zeros((len(data[:,0]), n))
for i in range(1,len(data[:,0])):
    x[i,:] = fx_fun(x[i-1, :], dt, data[i,1:])

#plot xsense
fig0, axes0 = plt.subplots(3,1)
fig0.suptitle("Xsense position reconstruction")
axes0[0].plot(x[:,1]); axes0[0].set_ylabel("x")
axes0[1].plot(x[:,2]); axes0[1].set_ylabel("y")
axes0[2].plot(x[:,3]); axes0[2].set_ylabel("z")

plt.show()