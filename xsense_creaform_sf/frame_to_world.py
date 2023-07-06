import numpy as np
from scipy.spatial.transform import Rotation as R
from numpy import cos 
from numpy import sin 

def mat_EE_to_xyz(alpha = np.radians(180), beta= np.radians(0), gamma= np.radians(135), a=(28.*0.5-7.6)*10.**(-3), b=-(31.5*0.5-7.6)*10.**(-3), c=7.24*10.**(-3)):
    #Translation and rotation matrix from EE to xyz
    rot_x = [[1, 0, 0, 0],
             [0, cos(alpha), -sin(alpha), 0],
             [0, sin(alpha), cos(alpha), 0],
             [0, 0, 0, 1]]
    rot_y = [[cos(beta), 0, sin(beta), 0],
             [0, 1, 0, 0],
             [-sin(beta), 0, cos(beta), 0],
             [0, 0, 0, 1]]
    rot_z = [[cos(gamma), -sin(gamma), 0, 0],
             [sin(gamma), cos(gamma), 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]]
    trans_x = np.eye(4); trans_x[0,3] = a
    trans_y = np.eye(4); trans_y[1,3] = b
    trans_z = np.eye(4); trans_z[2,3] = c

    tr_EE_to_xyz = rot_x * trans_x * trans_y * trans_z * rot_z 

    return tr_EE_to_xyz

def mat_W_to_EE(RPY_UR, trans_UR):
    #Get the RPY and translation from ROS and build the matrix
    tr_W_to_EE = 1

    return tr_W_to_EE

def align_to_W(roll, pitch, yaw, RPY_UR=[0,0,0], trans_UR=[0,0,0]):
    #INPUT the euler angle array to rotate to the EE frame
    RPY = np.radians([roll,pitch,yaw])
    R_xyz_to_ENU = R.from_euler('xyz', RPY).as_matrix()
    R_ENU_to_xyz = np.linalg.inv(R_xyz_to_ENU).astype('float64') #va da ENU ad XSens
    
    tr_ENU_to_xyz = np.hstack((np.vstack((R_ENU_to_xyz, [0,0,0])), np.array([[0],[0],[0],[1]]))) #create translation matrix fro enu to xyz (no translation effectively)
    tr_xyz_mod = mat_W_to_EE(RPY_UR, trans_UR) * mat_EE_to_xyz() * tr_ENU_to_xyz #PO matrix composition including the trasform to the world frame (the robot base)

    


    return tr_xyz_mod