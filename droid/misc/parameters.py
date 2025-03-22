import os
from cv2 import aruco

# Robot Params #
nuc_ip = '172.16.0.4'
robot_ip = '172.16.0.2'
laptop_ip = "127.0.0.1"
sudo_password = 'robotlearning'
robot_type = "panda"  # 'panda' or 'fr3'
robot_serial_number = ""

# Camera ID's #
hand_camera_id = '14436910'
varied_camera_1_id = '25455306'
varied_camera_2_id = '27085680'

# Charuco Board Params #
CHARUCOBOARD_ROWCOUNT = 9
CHARUCOBOARD_COLCOUNT = 12
CHARUCOBOARD_CHECKER_SIZE = 0.030
CHARUCOBOARD_MARKER_SIZE = 0.023
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_100)

# Ubuntu Pro Token (RT PATCH) #
ubuntu_pro_token = ""

# Code Version [DONT CHANGE] #
droid_version = "1.3"

