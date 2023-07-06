#!/usr/bin/env python
from rigid_trans_fitter3D import RigidTransFitter3D
import numpy as np
import matplotlib.pyplot as plt

'''cam3D = [
[0.505, -0.207, 1.228],
[0.507, -0.208, 1.232],
[0.506, -0.208, 1.231],
[0.504, -0.104, 1.176],
[0.503, -0.104, 1.175],
[0.503, -0.103, 1.173],
[0.506, -0.001, 1.283],
[0.505, -0.001, 1.281],
[0.505, -0.001, 1.281],
[0.407, -0.105, 1.298],
[0.407, -0.105, 1.297],
[0.406, -0.105, 1.295],
[0.405, -0.000, 1.186],
[0.405, -0.000, 1.186],
[0.405, -0.000, 1.188],
[0.305, -0.316, 1.256],
[0.306, -0.317, 1.259],
[0.306, -0.317, 1.259],
[0.301, -0.207, 1.187],
[0.301, -0.207, 1.189],
[0.301, -0.207, 1.188],
[0.303, -0.105, 1.246],
[0.303, -0.105, 1.246],
[0.303, -0.105, 1.245],
[0.198, 0.001 , 1.238],
[0.198, 0.001 , 1.238],
[0.198, 0.001 , 1.238],
[0.197, 0.106 , 1.181],
[0.197, 0.105 , 1.177],
[0.197, 0.106 , 1.178],
]
rob3D = [
[1.588, 0.507, 0.706],
[1.588, 0.507, 0.706],
[1.588, 0.507, 0.706],
[1.688, 0.506, 0.756],
[1.688, 0.506, 0.756],
[1.688, 0.506, 0.756],
[1.788, 0.506, 0.656],
[1.788, 0.506, 0.656],
[1.788, 0.506, 0.656],
[1.688, 0.406, 0.655],
[1.688, 0.406, 0.655],
[1.688, 0.406, 0.655],
[1.788, 0.405, 0.755],
[1.788, 0.405, 0.755],
[1.788, 0.405, 0.755],
[1.487, 0.307, 0.705],
[1.487, 0.307, 0.705],
[1.487, 0.307, 0.705],
[1.588, 0.307, 0.755],
[1.588, 0.307, 0.755],
[1.588, 0.307, 0.755],
[1.687, 0.307, 0.705],
[1.687, 0.307, 0.705],
[1.687, 0.307, 0.705],
[1.787, 0.206, 0.704],
[1.787, 0.206, 0.704],
[1.787, 0.206, 0.704],
[1.887, 0.205, 0.754],
[1.887, 0.205, 0.754],
[1.887, 0.205, 0.754],
]'''

def transformer(cam3D, rob3D):
    calibration = RigidTransFitter3D()
    tran_rob2cam = calibration.get_transform(cam3D, rob3D)  # check order for feedbing points
    #print('\ncalculating the transformation matrix:')
    #print(tran_rob2cam)

    rob3D_goal = []
    rob3D_g = []
    for idx,i in enumerate(cam3D):
        i.append(1)
        rob3D_goal.append(np.matmul(tran_rob2cam, np.transpose(i)))
        rob3D_g.append(np.matmul(tran_rob2cam, np.transpose(i))[0:3])

    errX = []
    errY = []
    errZ = []
    dist = []
    for idx, val in enumerate(rob3D):
        #print(val)
        #print(rob3D_g[idx])
        errX.append(val[0]-rob3D_goal[idx][0])
        errY.append(val[1]-rob3D_goal[idx][1])
        errZ.append(val[2]-rob3D_goal[idx][2])
        dist.append(np.linalg.norm(val-rob3D_g[idx]))

    #print(np.std(errX))
    #print(np.std(errY))
    #print(np.std(errZ))
    #print(np.std(dist))
    fig = plt.figure()
    plt.plot(errX,'r')
    plt.plot(errY,'b')
    plt.plot(errZ,'g')
    plt.plot(dist,'m')
    fig.legend(["x [m]", "y [m]", "z [m]", "dist [m]"])
    # plt.error('some numbers')
    #plt.show()

    return tran_rob2cam

'''print(type(cam3D))
print("\n", type(rob3D))

transformer(cam3D, rob3D)'''