data collection:

roslaunch ur_robot_driver ur10e_bringup.launch robot_ip:=192.168.56.11

roslaunch xsens_mti_driver xsens_mti_node.launch 

move to the directory ROS_files

rosbag record filter/free_acceleration imu/time_ref tf joint_states -O nameofoutputfile
python extract_csv.py

python rework_csv.py
