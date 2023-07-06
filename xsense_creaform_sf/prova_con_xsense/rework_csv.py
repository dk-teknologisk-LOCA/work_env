import numpy as np
import os, glob, sys, time
import pandas as pd
from subprocess import call

def read_data(file, column_list):
    df = pd.read_csv(file, usecols=column_list)
    data = pd.DataFrame(df)
    data = data[column_list]

    return data

call(["gnome-terminal", "-x", "sh", "-c", "roscore"])
time.sleep(2)
os.system("rosparam set use_sim_time true")

i=1
name_list = []
for file in sorted(glob.glob("*.bag")):
    print("elaborating ", file)
    output_file_name = "run_"+str(i)
    i = i+1
    topiclist = ["/filter/free_acceleration", "/filter/quaternion", "/imu/time_ref"]
    topicnames = ["free_acceleration", "quaternion", "imu_time_ref"]
    name = file[:-4]
    name_list.append(name)
    os.system("mkdir "+name)
    for j, topic in enumerate(topiclist):
        os.system("rostopic echo -p -b "+file+" "+topic+" > "+output_file_name+"_"+topicnames[j]+".csv")
        os.system("mv "+output_file_name+"_"+topicnames[j]+".csv "+name)
    os.system("mv "+file+" "+name)

time.sleep(2)

os.system("killall -9 roscore")
time.sleep(1)
os.system("killall -9 rosmaster")
time.sleep(2)

#reorganise the topics in an useful way
dir_runs = []
for dir in sorted(glob.glob("/home/lorenzo/arb_bwe/data_testing/sensor_fusion/prova_con_xsense/run*")): #modify the name list
    dir_runs.append(dir)

for i,dir in enumerate(dir_runs):
    #raw data extraction
    imu_reftime = read_data(dir+"/run_"+str(i+1)+"_imu_time_ref.csv", ["field.time_ref"])
    imu_acceleration = read_data(dir+"/run_"+str(i+1)+"_free_acceleration.csv", ["field.vector.x", "field.vector.y", "field.vector.z"])
    imu_rotation = read_data(dir+"/run_"+str(i+1)+"_quaternion.csv", ["field.quaternion.x", "field.quaternion.y", "field.quaternion.z", "field.quaternion.w"])
    #data re-elaboration
    imu_reftime = imu_reftime.rename(columns={"field.time_ref":"time_ref"})
    
    imu_acceleration['ts'] = imu_reftime.time_ref
    imu_acceleration = imu_acceleration.rename(columns={"field.vector.x":"fax", "field.vector.y":"fay", "field.vector.z":"faz"})

    imu_rotation['ts'] = imu_reftime.time_ref
    imu_rotation = imu_rotation.rename(columns={"field.quaternion.x":"ox", "field.quaternion.y":"oy", "field.quaternion.z":"oz", "field.quaternion.w":"ow"})
    imu_rotation = imu_rotation[["ts", "ox", "oy" ,"oz", "ow"]]

    imu_acceleration['ox'] = imu_rotation.ox
    imu_acceleration['oy'] = imu_rotation.oy
    imu_acceleration['oz'] = imu_rotation.oz
    imu_acceleration['ow'] = imu_rotation.ow
    
    imu_acceleration = imu_acceleration[["ts", "fax", "fay" ,"faz", "ox", "oy" ,"oz", "ow"]]

    namerun = os.path.basename(os.path.normpath(dir))
    imu_acceleration.to_csv("/home/lorenzo/arb_bwe/data_testing/sensor_fusion/prova_con_xsense/xsense/"+namerun+"_imu_acceleration.csv")

