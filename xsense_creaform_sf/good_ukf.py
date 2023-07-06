import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sg
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
from frame_to_world import *
import sys

n = 9
m = 3#6 #6 is if we take also the rotation from the Creaform camera

#states vector: sx, sy, sz, r, p, y, vx, vy, vz
#input vector: fax, fay, faz, wx, wy, wz

global he_calibmat, xsense_freq
he_calibmat = np.loadtxt("/home/lorenzo/arb_bwe/robotics/hand_eye_calibration/he_calibration_matrix.txt")
xsense_freq = 100
dt = 1./xsense_freq

#FUNCTIONS FOR THE UKF
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

def hx_fun(measurement):
    yout = np.zeros(m)
    yout[0] = measurement[0]
    yout[1] = measurement[1]
    yout[2] = measurement[2]
    
    #yout[3] = measurement[3]
    #yout[4] = measurement[4]
    #yout[5] = measurement[5]

    return yout

#WARNING: THESE FUNCTIONS ARE ONLY TO SHOW THE TRJECTORY COMPUTED ONLY WITH THE XSENSE, FOR COMPARISON. 
def manu_eq_traj(s_prev=0, v_prev=0, acc_prev=0, t=0):
    v_act = v_prev + acc_prev*t
    s = s_prev + v_prev*t+acc_prev*(t**2)*0.5
    
    return v_act, s 

def xsens_traj(data):
    #data is ts, fax, fay, faz
    data=np.hstack([data, np.zeros((len(data[:,0]), 6))])
    ####           0    1    2    3   4   5   6   7   8   9
    #the order is ts, fax, fay, faz, vx, vy, vz, sx, sy, sz
    ####CALCULATION OF POSITION WITH MOTION EQUATIONS
    for i in range(1, data.shape[0]):
        dt = 1/xsense_freq #data[i,0]*10.**(-9.)
        data[i,4:7] = data[i-1,4:7] + data[i,1:4]*dt
        data[i,7:] = data[i-1, 7:] + data[i-1,4:7] * dt + 0.5 * data[i,1:4] * dt ** 2

    return data

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

def load_creaform(filename):
    #checked, working
    global he_calibmat
    #rotation is fdiscarded from the creaform camera
    df = pd.read_csv(filename, skiprows=1, usecols=[0,1,2,3,7], header=0, names=['TS', 'TX', 'TY', 'TZ', 'Valid'])
    data = pd.DataFrame(df)
    data.drop(data[data.Valid == 0].index, inplace=True)
    #TSC, TX, TY, TZ, RX, RY, RZ, Valid
    data = data.to_numpy()[:,:-1]
    data[:,1:4] = data[:,1:4]/1000.
    #read the HE calibration matrix data and apply the rotation
    H_W_C = he_calibmat
    aug_pos_vec = np.hstack((data[:,1:], np.ones((data.shape[0],1)))).T
    data[:,1:] = np.matmul(H_W_C, aug_pos_vec)[:-1,:].T

    return data

def load_xsens(filename):
    #ts, fax, fay, faz for xsens, input vector in the prediction phase
    df = pd.read_csv(filename)
    data = pd.DataFrame(df)
    data = data[["ts", "fax", "fay", "faz"]]
    data = data.to_numpy()

    #plot xsense data pre filter
    #fig0, axes0 = plt.subplots(3,1)
    #fig0.suptitle("Xsense acc pre filter")
    #axes0[0].plot(data[:,1]); axes0[0].set_ylabel("x")
    #axes0[1].plot(data[:,2]); axes0[1].set_ylabel("y")
    #axes0[2].plot(data[:,3]); axes0[2].set_ylabel("z")

    #low pass filter for the acceeration of xsense. Logic is to eliminate the jerk assocated with the robot and, in future,
    # with the human operator.
    #acc_cut_off = 2
    #tuning variable depending on the platform used and on the acceleration/speed of the platform and motion themselves
    #acc_vec = np.sqrt(data[:,1]**2 + data[:,2]**2 + data[:,3]**2)
    #med_acc = np.mean(acc_vec)
    #acc_vec= acc_vec-med_acc
    #for i in range(len(acc_vec)):
    #    if np.abs(acc_vec[i]) < acc_cut_off:
    #        data[i,1:4] = np.zeros(data[i,1:4].shape)
    out = np.copy(data)
    for i in range(1,data.shape[1]):
        out[:,i] = filtering(data[:,i])

    #plot xsense data post filter
    #fig0, axes0 = plt.subplots(3,1)
    #fig0.suptitle("Xsense acc post filter")
    #axes0[0].plot(out[:,1]); axes0[0].set_ylabel("x")
    #axes0[1].plot(out[:,2]); axes0[1].set_ylabel("y")
    #axes0[2].plot(out[:,3]); axes0[2].set_ylabel("z")

    return data

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

def prepare_data():
    #xsens_data = load_xsens_from_mt_manager("/home/lorenzo/arb_bwe/data_testing/sensor_fusion/combined_data/xsens/xsens_run_1_100.csv")
    xsens_data = load_xsens("/home/lorenzo/arb_bwe/data_testing/sensor_fusion/combined_data/xsense/run_1_imu_acceleration.csv")
    robot_data = load_robot("/home/lorenzo/arb_bwe/data_testing/sensor_fusion/combined_data/UR/run_1_tf.csv")
    creaform_data = load_creaform("/home/lorenzo/arb_bwe/data_testing/sensor_fusion/combined_data/creaform/creaform_run_1.CSV")
    #xsens_data_rotated = rotate_xsense_to_W(xsens_data)
    sim_time = np.ceil(creaform_data[-1,0])
    camera_rate = 80
    xsens_rate = 100

    return sim_time, creaform_data, xsens_data, robot_data

def plot_linear(xsense_reco, c_data, truth, state_reco):
    init_x = 100
    #plot xsense
    fig0, axes0 = plt.subplots(3,1)
    fig0.suptitle("Xsense")
    axes0[0].plot(xsense_reco[init_x:,7]); axes0[0].set_ylabel("x")
    axes0[1].plot(xsense_reco[init_x:,8]); axes0[1].set_ylabel("y")
    axes0[2].plot(xsense_reco[init_x:,9]); axes0[2].set_ylabel("z")

    #plot creaform
    fig1, axes1 = plt.subplots(3,1)
    fig1.suptitle("Creaform")
    axes1[0].plot(c_data[init_x:,1]); axes1[0].set_ylabel("x")
    axes1[1].plot(c_data[init_x:,2]); axes1[1].set_ylabel("y")
    axes1[2].plot(c_data[init_x:,3]); axes1[2].set_ylabel("z")

    #plot UR
    fig2, axes2 = plt.subplots(3,1)
    fig2.suptitle("Robot")
    axes2[0].plot(truth[init_x:,0]); axes2[0].set_ylabel("x")
    axes2[1].plot(truth[init_x:,1]); axes2[1].set_ylabel("y")
    axes2[2].plot(truth[init_x:,2]); axes2[2].set_ylabel("z")

    #plot sensor fusion
    fig3, axes3 = plt.subplots(3,1)
    fig3.suptitle("Sensor fusion")
    axes3[0].plot(state_reco[init_x:,0]); axes3[0].set_ylabel("x")
    axes3[1].plot(state_reco[init_x:,1]); axes3[0].set_ylabel("y")
    axes3[2].plot(state_reco[init_x:,2]); axes3[0].set_ylabel("z")

def plot_3d(xsense_reco, c_data, truth, state_reco):
    fig= plt.figure()
    ax = fig.add_subplot(111, projection='3d') 
    ax.plot3D(xsense_reco[:,7], xsense_reco[:,8], xsense_reco[:,9]), ax.set_title("xsense")
    ax.scatter(xsense_reco[0,7], xsense_reco[0,8], xsense_reco[0,9], color="red")
    ax.set_xlabel("z")
    ax.set_ylabel("x")
    ax.set_zlabel("y")

    fig1= plt.figure() 
    ax1 = fig1.add_subplot(111, projection='3d') 
    ax1.plot3D(c_data[:,0], c_data[:,1], c_data[:,2]), ax1.set_title("creaform")
    ax1.scatter(c_data[0,0], c_data[0,1], c_data[0,2], color="red")
    ax1.set_xlabel("z")
    ax1.set_ylabel("x")
    ax1.set_zlabel("y")

    fig2= plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d') 
    ax2.plot3D(truth[:,0], truth[:,1], truth[:,2]), ax2.set_title("truth")
    ax2.scatter(truth[0,0], truth[0,1], truth[0,2], color="red")
    ax2.set_xlabel("z")
    ax2.set_ylabel("x")
    ax2.set_zlabel("y")

    fig3= plt.figure()
    ax3 = fig3.add_subplot(111, projection='3d') 
    ax3.plot3D(state_reco[:,0], state_reco[:,1], state_reco[:,2]), ax3.set_title("sensor_fusion")
    ax3.scatter(state_reco[0,0], state_reco[0,1], state_reco[0,2], color="red")
    ax3.set_xlabel("z")
    ax3.set_ylabel("x")
    ax3.set_zlabel("y")

sim_time, c_data, xsense_data, truth = prepare_data()
xsense_reco = xsens_traj(xsense_data)
  
points = MerweScaledSigmaPoints(n= n, alpha=.1, beta=2., kappa=-1)

UKF = UnscentedKalmanFilter(dim_x=n, dim_z=m, dt=dt, hx=hx_fun, fx=fx_fun, points=points)

UKF.x = np.zeros(n) #todo put initial position of xsense holder
UKF.P *= 0.2 #initial uncertainty
z_std = 0.1**2.
UKF.R = np.eye(m)*z_std
state_std = 0.01**2.
UKF.Q = np.eye(n)*state_std#Q_discrete_white_noise(dim=m, dt = dt, var = 0.01**2, block_size = 2)

state_reco = []
c = 0
k = 0
j = 0
fine = 0
while fine == 0:
    c = c + 0.25
    if c % 1 == 0:
        UKF.predict(u=xsense_data[k,1:])
        k = k + 1
    if c % 1.25 == 0:
        if c_data[j, -1] == 0.:
            j = j + 1
            continue
        UKF.update(c_data[j, 1:4])
        j = j + 1
    state_reco.append(UKF.x[0:6])
    #if j == len(c_data):
    if xsense_data[k,0] == xsense_data[-1, 0] or c_data[j,0]==c_data[-1,0]:
        fine = 1
    #print('log-likelihood', UKF.log_likelihood)
#print("j, k", j,k)
#print("size(crea), size(xsense)", c_data.shape[0],xsense_data.shape[0])
state_reco = np.array(state_reco)

plot_linear(xsense_reco, c_data, truth, state_reco)
#
#plot_3d(xsense_reco, c_data, truth, state_reco)


plt.show()













'''
#WARNING: THESE FUNCTIONS ARE ONLY TO SHOW THE TRJECTORY COMPUTED ONLY WITH THE XSENSE, FOR COMPARISON. 
def load_xsens_from_mt_manager(filename):
    df = pd.read_csv(filename, skiprows=12)
    data = pd.DataFrame(df)[['SampleTimeFine', 'FreeAcc_E', 'FreeAcc_N', 'FreeAcc_U', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']]
    #TSX, FACCX, FACCY, FACCZ, WX, WY, WZ
    data["vx"]=0; data["vy"]=0; data["vz"]=0
    data["sx"]=0; data["sy"]=0; data["sz"]=0

    xsens_traj(data[["FreeAcc_E", "FreeAcc_N", 'FreeAcc_U', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', "sx", "sy", "sz", "vx", "vy", "vz"]])

    data = data[['SampleTimeFine', "sx", "sy", "sz", "vx", "vy", "vz", 'FreeAcc_E', 'FreeAcc_N', 'FreeAcc_U', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']]
    data = data.to_numpy()
    return data

def manu_eq_traj(s_prev=0, v_prev=0, acc_prev=0, t=0):
    v_act = v_prev + acc_prev*t
    s = s_prev + v_prev*t+acc_prev*(t**2)*0.5
    
    return v_act, s 

def xsens_traj(data):
    ####              0 1 2 3  4   5   6  7  8  9  10 11 12
    ####The order is ts R P Y FAX FAY FAZ vx vy vz sx st sz  for data
    data["R"]=0; data["P"]=0; data["Y"]=0; data["ts"]=0
    data = data[["ts", "R", "P", "Y","FreeAcc_E", "FreeAcc_N", 'FreeAcc_U', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', "sx", "sy", "sz", "vx", "vy", "vz"]]
    data = data.to_numpy()   ####CALCULATION OF POSITION WITH MOTION EQUATIONS
    for i in range(1, data.shape[0]):
        dt = 1/100. #data[i,0]*10.**(-9.)
        data[i,7], data[i,10] = manu_eq_traj(data[i-1,10], data[i-1,7], data[i-1,4], dt)
        data[i,8], data[i,11] = manu_eq_traj(data[i-1,11], data[i-1,8], data[i-1,5], dt)
        data[i,9], data[i,12] = manu_eq_traj(data[i-1,12], data[i-1,9], data[i-1,6], dt)
    
    fig1 = plt.figure()
    fig1.suptitle("Trajectory by XSense")
    ax1 = fig1.add_subplot(projection='3d')
    ax1.plot(data[:,10], data[:,11], data[:,12])
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_zlabel("z [m]")
    
    fig2 = plt.figure()
    fig2.suptitle("XYZ by Equations")
    ax1 = fig2.add_subplot(3,3,1)
    ax1.plot(data[:,10])
    ax2 = fig2.add_subplot(3,3,4)
    ax2.plot(data[:,11])
    ax3 = fig2.add_subplot(3,3,7)
    ax3.plot(data[:,12])

    return fig1, fig2

'''



'''def target(x):
    return np.exp(-(x - 2)**2) + np.exp(-(x - 6)**2/10) + 1/ (x**2 + 1)

def create_data():
    sim_time = 1
    n_samples = 100
    sample_rate = n_samples/sim_time
    t_sig = np.arange(0, sim_time, 1/n_samples)
    camera_rate = 80 #hz
    xsens_rate = 100 #hz

    camera_vec = []; xsens_vec = []
    for i in range(len(t_sig)):
        if i % (sample_rate/camera_rate) == 0:
            camera_vec.append(t_sig[i])
        if i % (sample_rate/xsens_rate) == 0:
            xsens_vec.append(t_sig[i])
    camera_samples = np.asarray(camera_vec, dtype=float)#.reshape(len(camera_vec), -1)
    xsens_samples = np.asarray(xsens_vec, dtype=float)#.reshape(len(xsens_vec), -1)

    #the ground truth
    truemot = np.array([target(t_sig), target(t_sig), target(t_sig), target(t_sig), 
                target(t_sig), target(t_sig)]).T

    #downsample for the camera
    truemot_downs = []
    j=0
    for i in range(len(t_sig)):
        if j < camera_samples.shape[0]:
            if t_sig[i] == camera_samples[j]:
                truemot_downs.append(truemot[i,:])
                j = j+1
    truemot_downs = np.asarray(truemot_downs, dtype=object)

    #camera measurements
    standardDev_cam = 0.01
    noise_cam = np.random.normal(0, standardDev_cam, len(camera_samples))
    zs_cam = np.array([truemot_downs[:,0] + noise_cam, truemot_downs[:,1] + noise_cam, truemot_downs[:,2] + noise_cam, truemot_downs[:,2] + noise_cam, 
                truemot_downs[:,4] + noise_cam, truemot_downs[:,5] + noise_cam])
    zs_cam = zs_cam.T


    standardDev_xsens = 0.5
    noise_xsens = np.random.normal(0, standardDev_xsens, len(xsens_samples))
    zs_xsens = np.array([target(xsens_samples) + noise_xsens, target(xsens_samples) + noise_xsens, target(xsens_samples) + noise_xsens, target(xsens_samples) + noise_xsens, 
                target(xsens_samples) + noise_xsens, target(xsens_samples) + noise_xsens])
    zs_xsens = zs_xsens.T

    return truemot, t_sig, zs_xsens, zs_cam, camera_samples'''

"""#old function for xsense loading data
def load_xsens(filename):
    #FAX, FAY, FAZ, WX, WY, WZ, Ox, Oy, Oz, Ow for xsens, input vector in the prediction phase
    global xb_calibmat
    
    H_W_NED = xb_calibmat

    df = pd.read_csv(filename)
    data = pd.DataFrame(df)
    data = data[["ts", "fax", "fay", "faz", "wx", "wy", "wz"]]
    data = data.to_numpy()
    #print(np.hstack([data[:,1:4], np.ones(data[:,0].shape).reshape(len(data[:,0]), 1)]))
    #print("\n", H_W_NED)
    #rotate data according to the correct world frame, from the NED frame
    rotated_fa = np.matmul(H_W_NED, np.hstack([data[:,1:4], np.ones(data[:,0].shape).reshape(len(data[:,0]), 1)]).T)
    rotated_av = np.matmul(H_W_NED, np.hstack([data[:,4:], np.ones(data[:,0].shape).reshape(len(data[:,0]), 1)]).T)

    data[:,1:4] = rotated_fa[:-1,:].T
    data[:,4:] = rotated_av[:-1,:].T

    return data"""


"""#old fx_fun
def fx_fun(x, dt, u):
    #states vector: sx, sy, sz, r, p, y, vx, vy, vz, abx, aby, abz
    #input vector: fax, fay, faz, wx, wy, wz
    
    #global init_FLAG, he_calibmat
    #if init_FLAG == False:
    #    rot_0 = he_calibmat
    #else:
    #    rot_0 = np.eye(3)

    xout= np.zeros(n)

    xout[0] = x[0] + x[6] * dt + 0.5 * x[9] * dt ** 2
    xout[1] = x[1] + x[7] * dt + 0.5 * x[10] * dt ** 2
    xout[2] = x[2] + x[8] * dt + 0.5 * x[11] * dt ** 2

    xout[6] = x[6] + x[9] * dt
    xout[7] = x[7] + x[10] * dt
    xout[8] = x[8] + x[11] * dt

    omegavec = np.array([u[3], u[4], u[5]]) ##[wx, wy, wz]
    if np.linalg.norm(omegavec) == 0.:
        omegaquat = np.hstack((np.array([0, 0, 0]), 1))
    else:
        #print("omegavec\n", omegavec)
        omegaquat = np.hstack((omegavec/np.linalg.norm(omegavec) * np.sin(np.linalg.norm(omegavec)*0.5*dt), np.cos(np.linalg.norm(omegavec)*0.5*dt)))
            
    rotquat = rpy_to_quat(x[3],x[4],x[5])
    quat = quatmul(rotquat, omegaquat)

    xout[3], xout[4], xout[5] = quat_to_rpy(quat)

    acc_quat = np.array([0, u[0], u[1], u[2]]); g_quat = np.array([0,0,0,9.81])
    adj_acc = quatmul(quatmul(quat, acc_quat), quatconj(quat))-g_quat
    xout[9] = adj_acc[1]
    xout[10] = adj_acc[2]
    xout[11] = adj_acc[3]

    return xout"""
