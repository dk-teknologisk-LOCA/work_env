to perform he calib as of 15/03/2023:

UR:
connect laptp to UR
MAKE SURE UR IS SET TO  TCP_POINTER (aka the tool is set directl on the flange)
run roslaunch ur_robot_driver ur10e_bringup.launch robot_ip:=192.168.56.10

Creaform:
Start the program, load the model, MAKE SURE ITS 80HZ and CENTERED ON TARGET.

collect data by starting the creaform and running "rosbag record /tf -O run_[number of run]"

note that it is not strictly necessary to run the two data collections at the same exact time, as they are filtered by mean/average


run extract_points_he_calib.py in the data/UR folder, this extracts the points from the bag files from the UR ROS files and saves them in separate folders by runs.
run data_extractor_he_calib.py in the "data" directory, this collects points from UR and from Creaform and saves ithem in a csv to be used for the camera calibration

the hand_eye_calibration_matri is saved into "he_calibration_matrix.txt" in the same hand eye calibration folder




NOTE: if you need to add another frame, it must be added in the file /catkin_ws/src/universal_robots/ur_description/urdf/inc/ur_macro.xacro
