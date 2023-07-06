from numpy import *
from math import *
from scipy.spatial.transform import Rotation as R

def Rx(theta):
  return matrix([[ 1, 0           , 0           ],
                   [ 0, cos(theta),-sin(theta)],
                   [ 0, sin(theta), cos(theta)]])
  
def Ry(theta):
  return matrix([[ cos(theta), 0, sin(theta)],
                   [ 0           , 1, 0           ],
                   [-sin(theta), 0, cos(theta)]])
  
def Rz(theta):
  return matrix([[ cos(theta), -sin(theta), 0 ],
                   [ sin(theta), cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])

def rotationMatrixToEulerAngles(R):
    sy = sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    x = atan2(R[2,1] , R[2,2])
    y = atan2(-R[2,0], sy)
    z = atan2(R[1,0], R[0,0])
 
    return array([x, y, z])

true_base_ENU_rotvec = deg2rad([0.24, -0.13, 107.64])
true_base_ENU_RPY = deg2rad([0.03, -0.23, 107.64])
print("I want rotvec: ", true_base_ENU_rotvec, "\n")
pos_base_xsens_rotvec = deg2rad(array([64.33, -73.46, 84.57]))
pos_base_xsens_RPY = deg2rad(array([-73.02, -76.61, 168.70]))
pos_xsens_base_RPY = deg2rad(array([73.02, 76.61, -168.70]))
H_x_w = Rz(pos_xsens_base_RPY[2]) * Ry(pos_xsens_base_RPY[1]) * Rx(pos_xsens_base_RPY[0])

pos_enu_xsens_RPY = deg2rad(array([-86.1, 12.5, -9.7]))
pos_xsens_enu_RPY = deg2rad(array([86.1, -12.5, 9.7]))
H_enu_x = Rz(pos_enu_xsens_RPY[2]) * Ry(pos_enu_xsens_RPY[1]) * Rx(pos_enu_xsens_RPY[0])

print("Middle mats: ", H_x_w, " \n   ", H_enu_x, "\n")
 
H_enu_w = linalg.inv(H_enu_x * H_x_w) 
print("I obtain as rotmat:    ", H_enu_w, "\n")
euler_w_enu = rotationMatrixToEulerAngles(H_enu_w)
print("\n", rad2deg(euler_w_enu))
