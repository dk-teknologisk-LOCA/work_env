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

call(["gnome-terminal", "-x", "sh", "-c", "roscore"])
time.sleep(3)
os.system("rosparam set use_sim_time true")

baglist = natsort.natsorted(pathlib.Path(__file__).parent.glob("*.bag"))
i=0
name_list = []
for file in baglist:
    print("Elaborating ", file)
    i = i+1
    output_file_name = "test_"+str(i)
    topiclist = ["/filter/free_acceleration"]
    topicnames = ["free_acceleration"]

    #check if directory for this test exists
    if (pathlib.Path(__file__).parent / output_file_name).exists() == 0:
        print("Creating directory for test ", i)
        (pathlib.Path(__file__).parent / output_file_name).mkdir()

    #exctract a csv for every topic of this test
    for j, topic in enumerate(topiclist):
        #extract the csv
        print("rostopic echo -p -b "+str(file)+" "+topic+" > "+str(file.parent)+"/"+output_file_name+"_"+topicnames[j]+".csv")
        os.system("rostopic echo -p -b "+str(file)+" "+topic+" > "+str(file.parent)+"/"+output_file_name+"_"+topicnames[j]+".csv")
        #move the csv to the correct directory
        os.rename(str(file.parent)+"/"+output_file_name+"_"+topicnames[j]+".csv", str(file.parent) + "/" + output_file_name+"/"+output_file_name+"_"+topicnames[j]+".csv")        

    #once the current test has been elaborated, the bag file is moved to the correspondent directory
    os.rename(str(file), str(file.parent) + "/" + output_file_name+"/"+str(file.name))

time.sleep(1)

#reorganise the topics in an useful way
for i,dir in enumerate(natsort.natsorted(pathlib.Path(__file__).parent.glob("test*"))):
    #raw data extraction, for cycle needed to let the generator generate the proper file path
    for file in dir.glob("*.csv"):
        continue
    #raw data extraction
    #imu_reftime = read_data(dir+"/test_"+str(i+1)+"_imu_time_ref.csv", ["field.time_ref"])
    imu_acceleration = read_data(file, ["field.vector.x", "field.vector.y", "field.vector.z"])
    #joint_states = read_data(dir+"/test_"+str(i+1)+"_joint_states.csv", ["field.position0", "field.position1", "field.position2", "field.position3", "field.position4", "field.position5", 
                                                                        #"field.velocity0", "field.velocity1", "field.velocity2", "field.velocity3", "field.velocity4", "field.velocity5"])
    #UR_free_acceleration = read_data(dir+"/test_"+str(i+1)+"_free_acceleration.csv", ["field.transforms0.header.frame_id", "field.transforms0.child_frame_id", "field.transforms0.transform.translation.x", "field.transforms0.transform.translation.y", "field.transforms0.transform.translation.z", "field.transforms0.transform.rotation.x", "field.transforms0.transform.rotation.y", "field.transforms0.transform.rotation.z", "field.transforms0.transform.rotation.w"])
    
    #data re-elaboration
    #imu_reftime = imu_reftime.rename(columns={"field.time_ref":"time_ref"})
    
    #imu_acceleration['ts'] = imu_reftime.time_ref
    imu_acceleration['ts'] = np.arange(0, imu_acceleration.shape[0]/80., 1./80.)
    imu_acceleration = imu_acceleration.rename(columns={"field.vector.x":"fax", "field.vector.y":"fay", "field.vector.z":"faz"})
    imu_acceleration = imu_acceleration[["ts", "fax", "fay" ,"faz"]]
    
    #joint_states = joint_states.rename(columns={"field.position0":"pos0", "field.position1":"pos1", "field.position2":"pos2", "field.position3":"pos3", "field.position4":"pos4", "field.position5":"pos5", 
    #                                            "field.velocity0":"vel0", "field.velocity1":"vel1", "field.velocity2":"vel2", "field.velocity3":"vel3", "field.velocity4":"vel4", "field.velocity5":"vel5"})
    
    #UR_free_acceleration = UR_free_acceleration.rename(columns={"field.transforms0.header.frame_id":"parent_frame", "field.transforms0.child_frame_id":"child_frame", "field.transforms0.transform.translation.x":"posx", "field.transforms0.transform.translation.y":"posy", "field.transforms0.transform.translation.z":"posz", "field.transforms0.transform.rotation.x":"rotx", "field.transforms0.transform.rotation.y":"roty", "field.transforms0.transform.rotation.z":"rotz", "field.transforms0.transform.rotation.w":"rotw"})
    
    #imu_reftime.to_csv("/home/lorenzo/arb_bwe/data_testing/sensor_fusion/combined_data/xsens/test_"+str(i+1)+"_imu_time_ref.csv")
    imu_acceleration.to_csv(file)
    #joint_states.to_csv("/home/lorenzo/arb_bwe/data_testing/sensor_fusion/combined_data/UR/test_"+str(i+1)+"_joint_states.csv")
    #UR_free_acceleration.to_csv("/home/lorenzo/arb_bwe/data_testing/sensor_fusion/combined_data/UR/test_"+str(i+1)+"_free_acceleration.csv")

