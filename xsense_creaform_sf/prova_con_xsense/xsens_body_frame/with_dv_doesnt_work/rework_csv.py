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
time.sleep(4)
os.system("rosparam set use_sim_time true")

i=1
name_list = []
for file in glob.glob("*.bag"):
    print(file)
    print("insert output name")
    output_file_name = "run_"+str(i)
    i = i+1
    topiclist = ["/imu/dv", "/imu/time_ref"]
    topicnames = ["dv", "imu_time_ref"]
    name = file[:-4]
    os.system("mkdir "+name)
    for j, topic in enumerate(topiclist):
        print("rostopic echo -p -b "+file+" "+topic+" > "+name+"/"+output_file_name+"_"+topicnames[j]+".csv")
        os.system("rostopic echo -p -b "+file+" "+topic+" > "+output_file_name+"_"+topicnames[j]+".csv")
        os.system("mv "+output_file_name+"_"+topicnames[j]+".csv "+name)
    os.system("mv "+file+" "+name)

time.sleep(5)

os.system("killall -9 roscore")
time.sleep(1)
os.system("killall -9 rosmaster")
time.sleep(5)

#reorganise the topics in an useful way
dir_runs = []
for dir in glob.glob("/home/lorenzo/arb_bwe/data_testing/sensor_fusion/prova_con_xsense/with_dv/20*"):
    dir_runs.append(dir)
dir_runs.reverse()
print(dir_runs)
for i,dir in enumerate(dir_runs):
    #raw data extraction
    imu_reftime = read_data(dir+"/run_"+str(i+1)+"_imu_time_ref.csv", ["field.time_ref"])
    imu_dv = read_data(dir+"/run_"+str(i+1)+"_dv.csv", ["field.vector.x", "field.vector.y", "field.vector.z"])
    
    #data re-elaboration
    imu_reftime = imu_reftime.rename(columns={"field.time_ref":"time_ref"})
    
    imu_dv['ts'] = imu_reftime.time_ref
    imu_dv = imu_dv.rename(columns={"field.vector.x":"vx", "field.vector.y":"vy", "field.vector.z":"vz"})
    imu_dv = imu_dv[["ts", "vx", "vy" ,"vz"]]
    
    
    #imu_reftime.to_csv("/home/lorenzo/arb_bwe/data_testing/sensor_fusion/combined_data/xsens/run_"+str(i+1)+"_imu_time_ref.csv")
    imu_dv.to_csv("/home/lorenzo/arb_bwe/data_testing/sensor_fusion/prova_con_xsense/with_dv/xsense/run_"+str(i+1)+"_imu_dv.csv")

