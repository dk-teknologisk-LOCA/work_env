import pandas as pd
from hwu_code import *
import numpy as np
import pathlib

def obtain_trans_matrix(number):
    points = pd.read_csv("he_calibration_data.csv", usecols=[1,2,3,4,5,6])
    points = pd.DataFrame(points)
    points = points.to_numpy()

    #Extract points from creaform
    creaform_points = points[:,:3]
    #Extract end effector points
    robot_points = points[:,3:]
    #Add the offset (matrix Hee_tool) becuse the camera doesn't see the end effector, but the tip of the tool
    #This translates in adding a translation of [0, 0, height of tool] and no rotation (because i decide to have the tool frame not rotated with respect to the ee frame)
    robot_points[:,-1] = robot_points[:,-1]+31.03*10**(-3.)

    cam3D = creaform_points
    rob3D = robot_points

    print(cam3D)
    print("\n", rob3D)

    trans_matrix = transformer(cam3D.tolist(), rob3D.tolist())

    return trans_matrix
 
tm = obtain_trans_matrix(number = -1)

with open(pathlib.Path(__file__).parent / "he_calibration_matrix.txt",'wb') as f:
    np.savetxt(f, tm, fmt='%.6f')

print("Hand_eye calib matrix: \n", tm)
plt.show()


