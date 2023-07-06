from numpy import *
import pandas as pd

"""
Frames: 
    S: xsens onboard body frame
    ENU: frame aligned to Earth with E = x, N = y and U = z
    Tool base: Task frame. Currently x is aligned with the long side of thhe table and z points upwards
    Tool frame: algned with xsens. Fixed on the tool's center

All units in radians    
"""

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



def get_initial_TB_x():
    """
    Initial orientation of xsens S frame wrt TOOL BASE (TB-->xS)
    This rotation is fixed as long as the tool shape doesn't change
    """
                        #w    x      y       z
    quat_TB_xS = array([0.0, 0.7071, 0.7071, 0.0])

    return quat_TB_xS

def get_ENU_TB(data):
    """
    Fixed orientation of the tool base wrt to ENU frame.
    For later use the rotation is inverted.
    INPUT:  one reading (or average of) of the orientation data from xsens while sitting at the TB
            type: quaternion
    OUTPUT: rotation matrix of TB wrt ENU (TB-->ENU)
    """
    quat_TB_xS = get_initial_TB_x()
    quat_xS_ENU = data
    quat_TB_ENU = quaternion_multiply(quat_TB_xS, quat_xS_ENU)
    quat_ENU_TB = invert_quat(quat_TB_ENU)

    return quat_ENU_TB

def rotate_data_wrt_TB(datapoint, quat_ENU_TB):
    """
    Function to rotate single data point to the TB perspective (aka the task frame perspective)
    INPUT:  - data point in the xS frame wrt to ENU [quaternion]. 
            - fixed rotation between ENU and TB (ENU-->TB)
    OUTPUT: data point in thhe xS frame wrt to TB. Orientation data wrt to the task frame
    """
    quat_xS_TB = quaternion_multiply(datapoint, quat_ENU_TB)

    return quat_xS_TB

def rotate_xsens_orientation(data):
    #load rotation matrix between ENU and TB
    quat_ENU_TB = get_ENU_TB(data.iloc[0][1:])
    rotated_data = pd.DataFrame(index=range(data.shape[0]), columns=["timestamp","qw","qx","qy","qz"])
    #rotate all data
    for i in range(data.shape[0]):
        rotated_data.iloc[i][1:]= rotate_data_wrt_TB(data.iloc[i][1:], quat_ENU_TB)
    rotated_data["timestamp"] = data["timestamp"]

    return rotated_data


