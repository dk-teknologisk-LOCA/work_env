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
    output_file_name = "run_tf_"+str(i)
    topiclist = ["/tf"]
    topicnames = ["tf"]

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
for i,dir in enumerate(natsort.natsorted(pathlib.Path(__file__).parent.glob("run_tf*"))):
    print("i, dir ", i, dir)
    #raw data extraction, for cycle needed to let the generator generate the proper file path
    for file in dir.glob("*.csv"):
        continue
    UR_tf = read_data(file, ["field.transforms0.header.frame_id", "field.transforms0.child_frame_id", "field.transforms0.transform.translation.x", "field.transforms0.transform.translation.y", "field.transforms0.transform.translation.z", "field.transforms0.transform.rotation.x", "field.transforms0.transform.rotation.y", "field.transforms0.transform.rotation.z", "field.transforms0.transform.rotation.w"])
    #data re-elaboration
    UR_tf = UR_tf.rename(columns={"field.transforms0.header.frame_id":"parent_frame", "field.transforms0.child_frame_id":"child_frame", "field.transforms0.transform.translation.x":"posx", "field.transforms0.transform.translation.y":"posy", "field.transforms0.transform.translation.z":"posz", "field.transforms0.transform.rotation.x":"rotx", "field.transforms0.transform.rotation.y":"roty", "field.transforms0.transform.rotation.z":"rotz", "field.transforms0.transform.rotation.w":"rotw"})
    UR_tf.drop(UR_tf[UR_tf.parent_frame != 'base'].index, inplace=True)
    UR_tf.to_csv(file, index=False)

