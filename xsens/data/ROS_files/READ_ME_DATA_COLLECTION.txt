----data collection:

if UR present:
    roslaunch ur_robot_driver ur10e_bringup.launch robot_ip:=192.168.56.11

roslaunch xsens_mti_driver xsens_mti_node.launch 

---- move to the directory ROS_files

rosbag record imu/time_ref /filter/quaternion filter/free_acceleration
    || if UR present:                                                                                                   ||
    ||      rosbag record imu/time_ref /filter/quaternion filter/free_acceleration tf joint_states -O nameofoutputfile  ||

python extract_*select_file*.py 