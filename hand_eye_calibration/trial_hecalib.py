import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

points = pd.read_csv("robotics/hand_eye_calibration/he_calibration_data.csv", usecols=[1,2,3,4,5,6])
points = pd.DataFrame(points)
points = points.to_numpy()

#Extract points from creaform
creaform_points = points[:,:3]
print("creaform points\n", creaform_points)
#Extract end effector points
robot_points = points[:,3:]
print("robot points\n", robot_points)
#Add the offset (matrix Hee_tool) becuse the camera doesn't see the end effector, but the tip of the tool
#This translates in adding a translation of [0, 0, height of tool] and no rotation (because i decide to have the tool frame not rotated with respect to the ee frame)
#robot_points[:,-1] = robot_points[:,-1]+31.03*10**(-3.)

cam3D = creaform_points
rob3D = robot_points

fig= plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d') 
ax.plot3D(creaform_points[:,0], creaform_points[:,1], creaform_points[:,2])
ax.set_xlabel("z")
ax.set_ylabel("x")
ax.set_zlabel("y")

fig2= plt.figure()
ax2 = fig2.add_subplot(1,1,1, projection='3d') 
ax2.plot3D(robot_points[:,0], robot_points[:,1], robot_points[:,2])
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("z")

plt.show()


