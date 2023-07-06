#
# This function handles the computatiojn of the transformation matrix from the NED (north-east-down) frame used by Xsense to measure the free acceleration to the body frame 
# a predefined world frame at the base of the robot manipulator, aka H_ned_w, this matrix can then be usd to convert all the ata from the xsens in the ned frame to the world 
# (robot base) frame. In equations:
#
# H_ned_body * H_body_xb * H_xb_w = H_ned_w
#
# H_w_ned * H_ned_a = H_ned_w^-1 * H_ned_a = H_w_a
#
# where: ned = north-east-down magnetic imu frame; body = imu frame on the body; xb = frame at the resting base of the tool (and of xsense); w = world frame, set at the 
# base of the robot; a = hypotetic point to be reached by the tool.
#
# Note 1: the matrix "H_ned_body" is equivalent to the initial readings of RPY, while the xsense is still in its resting pose, this can be calibrated once by knowing the pose 
# of the tool holder; also, the matrix H_body_xb and H_xb_w are also calibrated based on the tool holder pose.
#  
# Note 2: it is deliberate whether the rotation of H_body_xb or the rotation of H_xb_w should be 1. It is chosen to be 1 for the matrix H_body_xb.
#
# IMPORTANT: this means that the matrix H_ned_w is a propriety of the specific holder and depends only on its pose wrt to the W frame of reference
#

import numpy as np
from scipy.spatial.transform import Rotation

def extract_w_ned_mat():
    data_ned_body = np.loadtxt("robotics/hand_eye_calibration/rot_ned_body.txt") #this is rpy, column vector witht he angles
    r = Rotation.from_euler('zyx', data_ned_body, degrees=True)
    H_ned_body = np.zeros((4,4))
    H_ned_body[0:3, 0:3] = r.as_matrix(); H_ned_body[-1,:]=np.array([0,0,0,1]); H_ned_body[:-1, -1] = np.array([[0],[0],[0]]).reshape(3)
    print("H ned body\n", H_ned_body)

    data_body_xb = np.loadtxt("robotics/hand_eye_calibration/rot_body_xb.txt") #translation from body to xb, no rotation between the frames, column vector with the translation coordinates
    H_body_xb = np.array([[1, 0, 0, data_body_xb[0]], [0, 1, 0, data_body_xb[1]], [0, 0, 1, data_body_xb[2]], [0, 0, 0, 1]], )
    print("H body xb\n", H_body_xb)

    data_xb_w = np.loadtxt("robotics/hand_eye_calibration/rot_xb_w.txt") #rototranslation data from xb to w. vector wit xyz and quaternion terms
    r = Rotation.from_quat(data_xb_w[3:])
    H_xb_w = np.zeros((4,4)); H_xb_w[0:3, 0:3] = r.as_matrix(); H_xb_w[-1, :] = np.array([0,0,0,1]); H_xb_w[:-1, -1] = data_xb_w[:3]
    print("H xb w\n", H_xb_w)

    H_ned_w = np.matmul(np.matmul(H_ned_body, H_body_xb), H_xb_w)

    H_w_ned = np.linalg.inv(H_ned_w)

    return H_w_ned

calibmat = extract_w_ned_mat()
print("rotmat \n", calibmat)

with open('robotics/hand_eye_calibration/xsense_calibration_matrix.txt','wb') as f:
    np.savetxt(f, calibmat, fmt='%.2f')