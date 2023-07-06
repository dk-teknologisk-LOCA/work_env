import os, time, glob
from subprocess import call

call(["gnome-terminal", "-x", "sh", "-c", "roscore"])
time.sleep(4)
os.system("rosparam set use_sim_time true")

i=1
name_list = []
for file in sorted(glob.glob("*.bag"), key=os.path.getmtime):
    output_file_name = "run_"+str(i)
    print(" output name", output_file_name)
    i = i+1
    topiclist = ["/imu/acceleration", "/imu/time_ref", "/tf", "/joint_states"]
    topicnames = ["imu_acceleration", "imu_time_ref", "tf", "joint_states"]
    name = file[:-4]
    os.system("mkdir "+name)
    for j, topic in enumerate(topiclist):
        print("rostopic echo -p -b "+file+" "+topic+" > "+name+"/"+output_file_name+"_"+topicnames[j]+".csv")
        os.system("rostopic echo -p -b "+file+" "+topic+" > "+output_file_name+"_"+topicnames[j]+".csv")
        os.system("mv "+output_file_name+"_"+topicnames[j]+".csv "+name)
