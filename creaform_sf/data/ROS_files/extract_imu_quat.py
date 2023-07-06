import numpy as np
import os, glob, sys, time
import pandas as pd
from subprocess import call
import natsort, pathlib

def read_data(file, column_list):
    df = pd.read_csv(file, usecols=column_list)
    data = pd.DataFrame(df)
    data = data[column_list]

    return data

def filter_by_parent_frame(data):
    return data.drop(data[data.parent_frame != 'base'].index, inplace=True)

call(["gnome-terminal", "-x", "sh", "-c", "roscore"])
time.sleep(3)
os.system("rosparam set use_sim_time true")

baglist = natsort.natsorted(pathlib.Path(__file__).parent.glob("*.bag"))
i=0
name_list = []
for file in baglist:
    print("Elaborating ", file)
    i = i+1
    output_file_name = "run_quat_"+str(i)
    topiclist = ["/filter/quaternion"]
    topicnames = ["quaternion"]

    #check if directory for this run exists
    if (pathlib.Path(__file__).parent / output_file_name).exists() == 0:
        (pathlib.Path(__file__).parent / output_file_name).mkdir()

    #exctract a csv for every topic of this run
    for j, topic in enumerate(topiclist):
        #extract the csv
        os.system("rostopic echo -p -b "+str(file)+" "+topic+" > "+str(file.parent)+"/"+output_file_name+"_"+topicnames[j]+".csv")
        #move the csv to the correct directory
        os.rename(str(file.parent)+"/"+output_file_name+"_"+topicnames[j]+".csv", str(file.parent) + "/" + output_file_name+"/"+output_file_name+"_"+topicnames[j]+".csv")        

    #once the current run has been elaborated, the bag file is moved to the correspondent directory
    os.rename(str(file), str(file.parent) + "/" + output_file_name+"/"+str(file.name))

time.sleep(1)

#reorganise the topics in an useful way
for i,dir in enumerate(natsort.natsorted(pathlib.Path(__file__).parent.glob("run_quat*"))):
    print("i, dir ", i, dir)
    #raw data extraction, for cycle needed to let the generator generate the proper file path
    for file in dir.glob("*.csv"):
        continue
    IMU_quat = read_data(file, ["%time", "field.quaternion.w", "field.quaternion.x", "field.quaternion.y", "field.quaternion.z"])
    #data re-elaboration
    IMU_quat = IMU_quat.rename(columns={"%time":"timestamp", "field.quaternion.w":"qw", "field.quaternion.x":"qx", "field.quaternion.y":"qy", "field.quaternion.z":"qz"})
    #IMU_quat.drop(IMU_quat[IMU_quat.parent_frame != 'base'].index, inplace=True)
    IMU_quat.to_csv(file, index=False)

