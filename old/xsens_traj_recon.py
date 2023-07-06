from math import e
from re import A
from matplotlib.figure import figaspect
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
import math
from scipy.spatial.transform import Rotation as R
from data_testing.old.frame_to_world import *

test_run_name = {
    "x-": "x--000.csv",
    "x+": "x+-000.csv",
    "y-": "y--000.csv",
    "y+": "y+-000.csv",
    "zup": "zup-000.csv",
    "zdown": "zdown-000.csv",
    "bias": "biasdata.csv"
}

def get_xsens_data(filename):
    data = pd.read_csv(filename)
    df = pd.DataFrame(data)
    df['SampleTimeFine'] = (df['SampleTimeFine'].iloc[:]-df['SampleTimeFine'].iloc[0])*10**(-4)
    df.drop(labels=['PacketCounter', 'UTC_Nano', 'UTC_Year', 'UTC_Month', 'UTC_Day', 'UTC_Hour', 'UTC_Minute', 'UTC_Second', 'UTC_Valid'], axis = 1, inplace = True)
    df.drop(labels=['Mag_X', 'Mag_Y', 'Mag_Z'], axis = 1, inplace = True)
    df.dropna(axis='columns', inplace = True, how='all')
    df.dropna(inplace=True)

    return df

def plot_data_xsens(data, data_vel):
    RPY = data[['Roll', 'Pitch', 'Yaw']].to_numpy()
    ENU = data[['FreeAcc_E','FreeAcc_N','FreeAcc_U']].to_numpy()
    ts = data.index; ts = ts.to_numpy(); ts = ts.reshape(-1,1)
    print("RPY\n", ENU.shape)
    fig1 = plt.figure()
    fig1.suptitle("RPY data, ACC data and delta vs")
    ax1 = plt.subplot(3,3,1)
    ax1.plot(ts, RPY[:,0])
    ax1.legend("Roll")
    ax2 = plt.subplot(3,3,4)
    ax2.plot(ts, RPY[:,1])
    ax2.legend("Pitch")
    ax3 = plt.subplot(3,3,7)
    ax3.plot(ts, RPY[:,2])
    ax3.legend("Yaw")
    ax4 = plt.subplot(3,3,2)
    ax4.plot(ts, ENU[:,0])
    ax4.set_ylabel("E $[m/s^2]$")
    ax5 = plt.subplot(3,3,5)
    ax5.plot(ts, ENU[:,1])
    ax5.set_ylabel("N $[m/s^2]$")
    ax6 = plt.subplot(3,3,8)
    ax6.plot(ts, ENU[:,2])
    ax6.set_ylabel("U $[m/s^2]$")
    
    data_vel = data_vel.to_numpy()
    print("data_vel", data_vel.shape)
    ax7 = plt.subplot(3,3,3)
    ax7.plot(ts, data_vel[:, 4])
    ax7.set_xlabel("time")
    ax7.set_ylabel("dvx")
    ax8 = plt.subplot(3,3,6)
    ax8.plot(ts, data_vel[:, 5])
    ax8.set_xlabel("time")
    ax8.set_ylabel("dvy")
    ax9 = plt.subplot(3,3,9)
    ax9.plot(ts, data_vel[:, 6])
    ax9.set_xlabel("time")
    ax9.set_ylabel("dvz")

    return fig1

def manu_eq_traj(s_prev=0, v_prev=0, acc_prev=0, t=0):
    v_act = v_prev + acc_prev*t
    s = s_prev + v_prev*t+acc_prev*(t**2)*0.5
    
    return v_act, s 

def get_trajectory(data, data_vel):
    ####              0 1 2 3  4   5   6  7  8  9  10 11 12
    ####The order is ts R P Y FAX FAY FAZ vx vy vz sx st sz  for data
    ####              0   1   2   3   4   5   6  7   8  9
    ####The order is ts  FAX FAY FAZ dvx dvy dvz sx sy sz
    
    data = data.to_numpy()   ####CALCULATION OF POSITION WITH MOTION EQUATIONS
    data = data[:400, :]
    for i in range(1, data.shape[0]):
        dt = data[i,0]-data[i-1,0] #data[i,0]*10.**(-9.)
        data[i,7], data[i,10] = manu_eq_traj(data[i-1,10], data[i-1,7], data[i-1,4], dt)
        data[i,8], data[i,11] = manu_eq_traj(data[i-1,11], data[i-1,8], data[i-1,5], dt)
        data[i,9], data[i,12] = manu_eq_traj(data[i-1,12], data[i-1,9], data[i-1,6], dt)
    #data[:, 10], data[:, 11], data[:, 12] = trans_xyz_EE(data[:, 10], data[:, 11], data[:, 12]) #Translation

    integ_vx = [0]; integ_x = [0]
    integ_vy = [0]; integ_y = [0]
    integ_vz = [0]; integ_z = [0]
    for i in range(1, data[:,4].shape[0]): ####CALCULATION OF POSITION WITH POINTWISE INTEGRATION
        integ_vx.append(integ_vx[i-1] + scipy.integrate.romb(np.array([data[i-1,4], data[i,4]])))
        integ_vy.append(integ_vy[i-1] + scipy.integrate.romb(np.array([data[i-1,5], data[i,5]])))
        integ_vz.append(integ_vz[i-1] + scipy.integrate.romb(np.array([data[i-1,6], data[i,6]])))
    
    for i in range(1,len(integ_vx)):
        integ_x.append(integ_x[i-1]+scipy.integrate.romb(np.array([integ_vx[i-1], integ_vx[i]])))
        integ_y.append(integ_y[i-1]+scipy.integrate.romb(np.array([integ_vy[i-1], integ_vy[i]])))
        integ_z.append(integ_z[i-1]+scipy.integrate.romb(np.array([integ_vz[i-1], integ_vz[i]])))
    #integ_x, integ_y, integ_z = trans_xyz_EE(integ_x, integ_y, integ_z)

    #USING DELTA VS
    
    data_vel['VelInc_Z'] = data_vel['VelInc_Z'].iloc[:]-data_vel['VelInc_Z'].iloc[0]
    data_vel = data_vel.to_numpy()
    data_vel = data_vel[:400, :]
    for i in range(1, data_vel.shape[0]):
        dt =  (data_vel[i,0]-data_vel[i-1,0])
        data_vel[i, 7] = data_vel[i-1,7]+data_vel[i-1,4]*dt+0.5*data_vel[i-1,1]*dt**2
        data_vel[i, 8] = data_vel[i-1,8]+data_vel[i-1,5]*dt+0.5*data_vel[i-1,2]*dt**2 
        data_vel[i, 9] = data_vel[i-1,9]+data_vel[i-1,6]*dt+0.5*data_vel[i-1,3]*dt**2 
    
    #data_vel[:, 7], data_vel[:, 8], data_vel[:, 9] = trans_xyz_EE(data_vel[:, 7], data_vel[:, 8], data_vel[:, 9])

    fig1 = plt.figure()
    fig1.suptitle("Trajectory by Equations, Integration and Delta V")
    ax1 = fig1.add_subplot(1,3,1,projection='3d')
    ax1.plot(data[:,10], data[:,11], data[:,12])
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_zlabel("z [m]")
    ax2 = fig1.add_subplot(1,3,2,projection='3d')
    ax2.plot(integ_x, integ_y, integ_z)
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    ax2.set_zlabel("z [m]")
    ax3 = fig1.add_subplot(1,3,3,projection='3d')
    ax3.plot(data_vel[:, 7], data_vel[:, 8], data_vel[:,9])
    ax3.set_xlabel("x [m]")
    ax3.set_ylabel("y [m]")
    ax3.set_zlabel("z [m]")
    
    fig2 = plt.figure()
    fig2.suptitle("XYZ by Equations, Integrations and Delta V")
    ax1 = fig2.add_subplot(3,3,1)
    ax1.plot(data[:,10])
    ax2 = fig2.add_subplot(3,3,4)
    ax2.plot(data[:,11])
    ax3 = fig2.add_subplot(3,3,7)
    ax3.plot(data[:,12])
    ax4 = fig2.add_subplot(3,3,2)
    ax4.plot(integ_x)
    ax5 = fig2.add_subplot(3,3,5)
    ax5.plot(integ_y)
    ax6 = fig2.add_subplot(3,3,8)
    ax6.plot(integ_z)
    ax7 = fig2.add_subplot(3,3,3)
    ax7.plot(data_vel[:,7])
    ax8 = fig2.add_subplot(3,3,6)
    ax8.plot(data_vel[:,8])
    ax9 = fig2.add_subplot(3,3,9)
    ax9.plot(data_vel[:,9]) #data_vel[:,0]-data_vel[0,0]

    fig3 = plt.figure(figsize=figaspect(0.2))
    fig3.suptitle("Planar visualization of XYZ")
    ax1 = fig3.add_subplot(1,3,1)
    ax2 = fig3.add_subplot(1,3,2)
    ax3 = fig3.add_subplot(1,3,3)
    ax1.plot(integ_x, integ_y)
    ax1.plot(data[:,10], data[:,11])
    ax1.plot(data_vel[:, 7], data_vel[:, 8])
    ax1.legend(["integration", "equations", "V Deltas"])
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax2.plot(integ_y, integ_z)
    ax2.plot(data[:,11], data[:,12])
    ax2.plot(data_vel[:, 8], data_vel[:, 9])
    ax2.legend(["integration", "equations", "V Deltas"])
    ax2.set_xlabel("y")
    ax2.set_ylabel("z")
    ax3.plot(integ_x, integ_z)
    ax3.plot(data[:,10], data[:,12])
    ax3.plot(data_vel[:, 7], data_vel[:, 9])
    ax3.legend(["integration", "equations", "V Deltas"])
    ax3.set_xlabel("x")
    ax3.set_ylabel("z")

    return fig1, fig2, fig3

def dummy_data():
    data = pd.DataFrame()
    data['SampleTimeFine'] = np.linspace(0,60,1000)
    data['Roll']=0; data['Pitch'] = 0; data['Yaw']= 0
    data['FreeAcc_E'] = 1 #np.sin(data['SampleTimeFine'])
    data['FreeAcc_N'] = 0 #1e4*np.sin(data['SampleTimeFine'])
    data['FreeAcc_U'] = -9.81 #np.sin(data['SampleTimeFine'])

    print(data)
    data["vx"]=0; data["vy"]=0; data["vz"]=0
    data["sx"]=0; data["sy"]=0; data["sz"]=0

    data = data.to_numpy()

    for i in range(1, data.shape[0]):
        dt = data[i,0]-data[i-1,0] #0.0125 #(data[i,0] - data[i-1,0])*10.**(-3.) #*1e-3 sec 0.0025e3
        data[i,7], data[i,10] = manu_eq_traj(data[i-1,10], data[i-1,7], data[i-1,4], dt)
        data[i,8], data[i,11] = manu_eq_traj(data[i-1,11], data[i-1,8], data[i-1,5], dt)
        data[i,9], data[i,12] = manu_eq_traj(data[i-1,12], data[i-1,9], data[i-1,6], dt)

    print("data", data[:,10])
    fig1 = plt.figure()
    fig1.suptitle("Trajectory by Equations, Integration and Delta V")
    ax1 = fig1.add_subplot(projection='3d')
    ax1.plot(data[:,10], data[:,11], data[:,12])
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_zlabel("z [m]")

    plt.show()
    return data

def rotate_point(roll, pitch, yaw,  x_UR=0, y_UR=0, z_UR=0, roll_UR=0, pitch_UR=0, yaw_UR=0):
    RPY_UR = [roll_UR, pitch_UR, yaw_UR]
    trans_UR = [x_UR, y_UR, z_UR]
    rotmat = align_to_W(roll, pitch, yaw, RPY_UR, trans_UR)[0:3, 0:3]
    newangles = np.matmul(rotmat, np.array([[roll], [pitch], [yaw]])).reshape(3)
    return newangles #Should I ultiply by the point, then?

if __name__ == "__main__":
    ##Extract the whole data
    folder = "xsens_data"
    folderdocs = "/home/lorenzo/Documents/"
    print("x-, x+, y-, y+, zup, zdown or bias?")
    name = input()
    if name == 'bias':
        cutoff = 250000

    df = get_xsens_data(folderdocs+test_run_name[name])  #OBS: it also brings the sampletime back in second (it is sampled at 10k Hz)
    
    #Extract only RPY and Acc_X, Acc_Y and Acc_Z
    df_prim = df[['SampleTimeFine','Roll', 'Pitch', 'Yaw', 'FreeAcc_E','FreeAcc_N','FreeAcc_U']]
    df_prim["vx"]=0; df_prim["vy"]=0; df_prim["vz"]=0
    df_prim["sx"]=0; df_prim["sy"]=0; df_prim["sz"]=0
    #Rotate according to frame EE
    for i in range(len(df_prim["Roll"])):

        df_prim[["Roll", "Pitch", "Yaw"]].iloc[i] = rotate_point(df_prim["Roll"].iloc[i], df_prim["Pitch"].iloc[i], df_prim["Yaw"].iloc[i])

    df_vel = df[['SampleTimeFine', 'FreeAcc_E','FreeAcc_N','FreeAcc_U', 'VelInc_X', 'VelInc_Y','VelInc_Z']]
    df_vel["sx"]=0; df_vel["sy"]=0; df_vel["sz"]=0
    #dummy_data()

    print("df_prim", df_prim['SampleTimeFine'])
    fig1 = plot_data_xsens(df_prim, df_vel)
    fig2, fig3, fig4 = get_trajectory(df_prim, df_vel)
    plt.show()
    
