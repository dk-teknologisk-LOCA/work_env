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
    output_file_name = "run_full_"+str(i)
    topiclist = ["/imu/time_ref", "/filter/quaternion", "/filter/free_acceleration"]
    topicnames = ["time_ref", "quaternion", "free_acceleration"]

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
for i,dir in enumerate(natsort.natsorted(pathlib.Path(__file__).parent.glob("run_full_*"))):
    print("i, dir ", i, dir)
    #raw data extraction, for cycle needed to let the generator generate the proper file path
    for file in dir.glob("*.csv"):
        continue

    IMU_quat = read_data(str(file.parent) +"/"+output_file_name+"_"+topicnames[1]+".csv", ["field.quaternion.w", "field.quaternion.x", "field.quaternion.y", "field.quaternion.z"])
    IMU_acc = read_data(str(file.parent) +"/"+output_file_name+"_"+topicnames[2]+".csv", ["field.vector.x", "field.vector.y", "field.vector.z"])
    IMU_time = read_data(str(file.parent) +"/"+output_file_name+"_"+topicnames[0]+".csv", ["field.time_ref"])
    #data re-elaboration
    IMU_time = IMU_time.rename(columns={"field.time_ref":"time_ref"})
    IMU_quat = IMU_quat.rename(columns={"field.quaternion.w":"qw", "field.quaternion.x":"qx", "field.quaternion.y":"qy", "field.quaternion.z":"qz"})
    #IMU_quat["ts"] = IMU_time.time_ref; IMU_quat = IMU_quat[["ts", "qw", "qx", "qy", "qz"]]
    IMU_acc = IMU_acc.rename(columns={"field.vector.x":"fax", "field.vector.y":"fay", "field.vector.z":"faz"})
    #IMU_acc["ts"] = IMU_time.time_ref; IMU_acc = IMU_acc[["ts", "fax", "fay", "faz"]]
    
    #IMU_tot = pd.concat([IMU_time, IMU_quat, IMU_acc], axis=1, ignore_index=True, sort=False, names=["ts", "qw", "qx", "qy", "qz", "fax", "fay", "faz"])
    IMU_tot = IMU_time.join(IMU_quat.join(IMU_acc))
    IMU_time.to_csv(str(file.parent) +"/"+output_file_name+"_"+topicnames[0]+".csv", index=False)
    IMU_quat.to_csv(str(file.parent) +"/"+output_file_name+"_"+topicnames[1]+".csv", index=False)
    IMU_acc.to_csv(str(file.parent) +"/"+output_file_name+"_"+topicnames[2]+".csv", index=False)
    IMU_tot.to_csv(str(file.parent) +"/"+output_file_name+"_"+"full"+".csv", index=False)


