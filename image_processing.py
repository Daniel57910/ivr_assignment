#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError
from numpy.linalg import pinv
from copy import deepcopy
import os
from datetime import datetime
import time

from math import cos, degrees, acos

PIXEL_TO_CM = 3.225

def partial_x(angle_array):
    partial_theta_1 = 3 * np.cos(angle_array[0]) + 3 * np.cos(angle_array[0] + angle_array[1]) + 3 * np.cos(np.sum(angle_array))
    partial_theta_2 = 3* np.cos(angle_array[0] + angle_array[1]) + 3 * np.cos(np.sum(angle_array))
    partial_theta_3 = 3 * np.cos(np.sum(angle_array))

    return np.array([partial_theta_1, partial_theta_2, partial_theta_3])

def partial_y(angle_array):
    partial_theta_1 = -3 * np.sin(angle_array[0]) + -3 * np.sin(angle_array[0] + angle_array[1]) + -3 * np.sin(np.sum(angle_array))
    partial_theta_2 = -3* np.sin(angle_array[0] + angle_array[1]) + -3 * np.sin(np.sum(angle_array))
    partial_theta_3 = -3 * np.sin(np.sum(angle_array))

    return np.array([partial_theta_1, partial_theta_2, partial_theta_3])


def jacobian_matrix(angle_array):
    x_jacob = partial_x(angle_array)
    y_jacob = partial_y(angle_array)

    return np.stack([x_jacob, y_jacob])

def detect_end_effector(image):
    a = pixel2meter(image)
    endPos = a * (detect_yellow(image) - detect_red(image))
    return endPos

def pixel2meter(image):
    # Obtain the centre of each coloured blob
      circle1Pos = detect_blue(image)
      circle2Pos = detect_green(image)
      # find the distance between two circles
      dist = np.sum((circle1Pos - circle2Pos)**2)
      return 3 / np.sqrt(dist)

def detect_red(image):
    mask = cv2.inRange(image, (0, 0, 100), (0, 0, 255))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    M = cv2.moments(mask)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return np.array([cx, cy])

  # Detecting the centre of the green circle
def detect_green(image):
    mask = cv2.inRange(image, (0, 100, 0), (0, 255, 0))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    M = cv2.moments(mask)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return np.array([cx, cy])

  # Detecting the centre of the blue circle
def detect_blue(image):
    mask = cv2.inRange(image, (100, 0, 0), (255, 0, 0))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    M = cv2.moments(mask)
    try:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    except:
        cv2.imshow("mask", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return np.array([cx, cy])

  # Detecting the centre of the yellow circle
def detect_yellow(image):
    mask = cv2.inRange(image, (0, 100, 100), (0, 255, 255))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    M = cv2.moments(mask)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    return np.array([cx, cy])

def detect_joint_angles(image):
    a = pixel2meter(image)
    print(a)
    # Obtain the centre of each coloured blob
    center = a * detect_yellow(image)
    circle1Pos = a * detect_blue(image)
    circle2Pos = a * detect_green(image)
    circle3Pos = a * detect_red(image)
    # Solve using trigonometry
    ja1 = np.arctan2(center[0]- circle1Pos[0], center[1] - circle1Pos[1])
    ja2 = np.arctan2(circle1Pos[0]-circle2Pos[0], circle1Pos[1]-circle2Pos[1]) - ja1
    ja3 = np.arctan2(circle2Pos[0]-circle3Pos[0], circle2Pos[1]-circle3Pos[1]) - ja2 - ja1
    return np.array([ja1, ja2, ja3]), circle3Pos

def moment_identification(masked_image):
    moment = cv2.moments(masked_image)
    x, y = int(moment["m10"] / moment["m00"]), int(moment["m01"] / moment["m00"])
    return np.array([x, y])

def crop_around_template(link_blob_detect, link_config, template_shape):
    length, width = int(template_shape[0]/2), int(template_shape[1]/2)
    x, y = link_config['centre']
    y_min, y_max = y - length, y + length
    x_min, x_max = x - width, x + width

    cropped_image = link_blob_detect[y_min: y_max, x_min: x_max]
    cropped_image = cv2.bitwise_not(cropped_image)
    return cropped_image

def create_rotation_vector(link):
    bottom_x, top_x = link['bottom'][0], link['top'][0]
    rotation_matrix = np.arange(0, 181, 1)
    return rotation_matrix * -1 if bottom_x < top_x else rotation_matrix


def homogenous_transformation_2d(angles):
    return np.array([
        3*np.sin(angles[0]) + 3*np.sin(angles[0] + angles[1]) + 3*np.sin(np.sum(angles)),
        3*np.cos(angles[0]) + 3*np.cos(angles[0] + angles[1]) + 3*np.cos(np.sum(angles))
    ]).round(4)

def distance_transform(link_template, link_blob_detect, link_config, link_name, rotation_vector, current_time):

    # print(f"Cropping Link {link_name} with config: {link_config}")
    # print(f"Shape of template for cropping: {link_template.shape}")
    cropped_link = crop_around_template(link_blob_detect, link_config, link_template.shape)
    # cv2.imwrite(f"cropped_image_{link_name}_{current_time}.png", cropped_link)
    dist_transform = cv2.distanceTransform(cropped_link, cv2.DIST_L2, 0)
    """
    openCV get rotation matrix based on x, y whereas image/numpy based on y, x
    """
    rotation_matrix_shape = (
        int((link_template.shape[1] - 1) / 2),
        int((link_template.shape[0] - 1) / 2)
    )

    rotation_container = []
    for rot in rotation_vector:
        rotation_matrix = cv2.getRotationMatrix2D(
            rotation_matrix_shape,
            angle=rot,
            scale=1
        )

        target = cv2.warpAffine(
            src=link_template,
            M=rotation_matrix,
            dsize=(link_template.shape[1], link_template.shape[0])
        )

        distance = np.sum(target * dist_transform)

        rotation_container.append(
            (rot, distance)
        )

    rotation_container = np.array(rotation_container)
    result = rotation_container[rotation_container[:, 1].argsort()]
    return result[0][0]

def forward_kinematics_equation(angles):
    return


def debug_link(link_config, hsv):

    for blob in link_config:
        for coord in ["centre"]:
            pair = blob[coord]
            circle = cv2.circle(
                hsv,
                pair,
                3,
                (0, 0, 215),
                -1
            )


    # cv2.imwrite("bug_example.png", circle)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def find_countour(joint, type):
    contours, hierarchy = cv2.findContours(joint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = np.array(contours[0])
    contours = contours[:,0]
    if type == "bottom":
        x = contours[:,0].min()
        y = contours[contours[:,0] == x]
        y = y[:,1].max()
    else:
        x = contours[:, 0].max()
        y = contours[contours[:, 0] == x]
        y = y[:, 1].min()
    return (x, y)



def link_summary_stats(bottom_joint, top_joint):

    bottom_x, bottom_y = find_countour(bottom_joint, "bottom")
    top_x, top_y = find_countour(top_joint, "top")
    centroid_x, centroid_y = (
        int(top_x - ((top_x - bottom_x)/2)),
        int(top_y - ((top_y - bottom_y)/2)),
    )
    return {
        "bottom": (bottom_x, bottom_y),
        "centre": (centroid_x, centroid_y),
        "top": (top_x, top_y)
    }


def find_links(image_path):
    link= cv2.inRange(cv2.imread(image_path, 1), (200, 200, 200), (255, 255, 255))
    return link


def calculate_jacobian_inverse(baseline_joint_angles):
    jacobian = jacobian_matrix(baseline_joint_angles)
    pseudo_inverse = pinv(jacobian)
    print(f"Pseudo jacobian inverse = {pseudo_inverse}: shape {pseudo_inverse.shape}")
    return pseudo_inverse

class image_converter:

    # Defines publisher and subscriber
    def __init__(self):
        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()
        # initialize the node named image_processing
        rospy.init_node('image_processing', anonymous=True)
        # initialize a publisher to send messages to a topic named image_topic
        self.image_pub = rospy.Publisher("image_topic", Image, queue_size=1)
        # initialize a publisher to send joints' angular position to a topic called joints_pos
        self.joints_pub = rospy.Publisher("joints_pos", Float64MultiArray, queue_size=10)
        # initialize a subscriber to receive messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
        self.image_sub = rospy.Subscriber("/robot/camera1/image_raw", Image, self.callback)
        #(for lab3)
        # initialize a publisher to send robot end-effector position
        self.end_effector_pub = rospy.Publisher("end_effector_prediction", Float64MultiArray, queue_size=10)
        # initialize a publisher to send desired trajectory
        self.trajectory_pub = rospy.Publisher("trajectory", Float64MultiArray, queue_size=10)
        # initialize a publisher to send joints' angular position to the robot
        self.robot_joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
        self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
        self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
        # record the beginning time
        self.time_trajectory = rospy.get_time()
        self.current_time = rospy.get_time()

    # Define a circular trajectory (for lab 3)
    def trajectory(self):

        print(f"Time is: {rospy.get_time()}")
        # get current time
        cur_time = np.array([rospy.get_time() - self.time_trajectory])
        x_d = float(6 * np.cos(cur_time * np.pi / 100))
        y_d = float(6 + np.absolute(1.5 * np.sin(cur_time * np.pi / 100)))
        return np.array([x_d, y_d])

    # Receive data, process it, and publish
    def callback(self, data):
        # Receive the image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        """
        Returns binarization of image where 1 for colour match.
        Useful for moment identification as moments only captured around required colours, robot joints.
        """
        mask_yellow = cv2.inRange(hsv, (20, 100, 100), (30, 255, 255))
        mask_blue = cv2.inRange(hsv, (100,150,0), (140,255,255))
        mask_green = cv2.inRange(hsv, (36, 0, 0), (70, 255,255))
        mask_red = cv2.inRange(hsv, (0,50,20), (5,255,255))

        """
        Identifies mean/center of each blob
        """
        moments_yellow = moment_identification(mask_yellow)
        moments_blue = moment_identification(mask_blue)
        moments_green = moment_identification(mask_green)
        moments_red = moment_identification(mask_red)

        """
        Image on a 2d plane so distance between each moment is sqrt of squared sum.
        This is scaled by the fact that each pixel is 3.225cm
        """
        euclidean_scale= np.sum(np.square(moments_yellow - moments_blue))
        euclidean_scale = PIXEL_TO_CM / np.sqrt(euclidean_scale)
        # print(f"Euclidean Scale Green = {euclidean_scale}")

        moments_yellow = euclidean_scale * moments_yellow
        moments_blue = euclidean_scale * moments_blue
        moments_green = euclidean_scale * moments_green
        moments_red = euclidean_scale * moments_red

        """
        Using the moments we know the ratio but need the tangent of the blobs.
        Need to also substract distance already covered from starting point/yellow from successive blobs.
        """
        angle_yellow_blue = np.arctan2(moments_yellow[0] - moments_blue[0], moments_yellow[1] - moments_blue[1])
        angle_blue_green = np.arctan2(
            moments_blue[0] - moments_green[0],
            moments_blue[1] - moments_green[1]
        ) - angle_yellow_blue

        angle_green_red = np.arctan2(
            moments_green[0] - moments_red[0],
            moments_green[1] - moments_red[1]) - angle_blue_green -  angle_yellow_blue


        data_angles = np.array([angle_yellow_blue, angle_blue_green, angle_green_red])

        current_time = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        # cv2.imwrite(f"test_image_{current_time}.png", cv_image)

        # loading template for links as binary image (used in lab 2)
        self.link1 = cv2.inRange(cv2.imread('link1.png', 1), (200, 200, 200), (255, 255, 255))
        self.link2 = cv2.inRange(cv2.imread('link2.png', 1), (200, 200, 200), (255, 255, 255))
        self.link3 = cv2.inRange(cv2.imread('link3.png', 1), (200, 200, 200), (255, 255, 255))

        # Perform image processing task (your code goes here)
        link_blob = cv2.inRange(hsv, (0, 0, 0), (50, 50, 100))

        bottom_link = link_summary_stats(mask_yellow, mask_blue)
        middle_link = link_summary_stats(mask_blue, mask_green)
        top_link = link_summary_stats(mask_green, mask_red)

        rotation_vector_bottom = create_rotation_vector(bottom_link)
        rotation_vector_middle = create_rotation_vector(middle_link)
        rotation_vector_top = create_rotation_vector(top_link)

        link_degrees_bottom = distance_transform(
            self.link1,
            deepcopy(link_blob),
            bottom_link,
            "link_1",
            rotation_vector_bottom,
            current_time)

        link_degrees_middle = distance_transform(
            self.link2,
            deepcopy(link_blob),
            middle_link,
            "link_2",
            rotation_vector_middle,
            current_time)

        link_degrees_top = distance_transform(
            self.link3,
            deepcopy(link_blob),
            top_link,
            "link_3",
            rotation_vector_top,
            current_time)

        link_radians_bottom = np.radians(link_degrees_bottom)
        link_radians_middle = np.radians(link_degrees_middle) - link_radians_bottom
        link_radians_top = np.radians(link_degrees_top) - link_radians_middle - link_radians_bottom

        joint_array = np.array([link_radians_bottom, link_radians_middle, link_radians_top])
        baseline_joint_angles, circle3_pos = detect_joint_angles(cv_image)
        end_effector = detect_end_effector(cv_image)

        print(f"Blob angle array {current_time}: {data_angles}")
        print(f"Joint Angle array {current_time}: {joint_array}")
        print(f"Baseline truth array {current_time}: {baseline_joint_angles}")
        print(f"Baseline truth circle3 pos: {end_effector}")
        self.joints = Float64MultiArray()
        """
        Concatenate both, publish and compre against baseline
        """
        self.joints.data = np.concatenate([data_angles, joint_array, baseline_joint_angles], axis=None)

        """
        Lab part 3
        """
        x_d_old = self.trajectory()    # getting the desired trajectory
        print(f"Trajectory at current time {current_time}: {x_d_old}")
        self.trajectory_desired= Float64MultiArray()
        self.trajectory_desired.data=x_d_old
        current_end_effector_pos = homogenous_transformation_2d(baseline_joint_angles)
        print(f"Location Based on homogenous transformation: {current_end_effector_pos}")

        jacobian_inverse = calculate_jacobian_inverse(baseline_joint_angles)
        self.end_effector=Float64MultiArray()
        self.end_effector.data = current_end_effector_pos

        x_d_latest = self.trajectory()
        time.sleep(0.25)
        new_time =rospy.get_time()
        delta = (x_d_latest - x_d_old) / (new_time - self.current_time)
        self.current_time = new_time
        print(f"Delta = {delta}")

        angle_update = jacobian_inverse @ delta
        q_d = baseline_joint_angles + angle_update
        print(f"Angle Update = {angle_update}")

        # send control commands to joints (for lab 3)
        self.joint1=Float64()
        self.joint1.data= q_d[0]
        self.joint2=Float64()
        self.joint2.data= q_d[1]
        self.joint3=Float64()
        self.joint3.data= q_d[2]

        # Publishing the desired trajectory on a topic named trajectory(for lab 3)

        self.trajectory_desired= Float64MultiArray()
        self.trajectory_desired.data=x_d_old

        # Publish the results - the images are published under a topic named "image_topic" and calculated joints angles are published under a topic named "joints_pos"
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
            self.joints_pub.publish(self.joints)
            # (for lab 3)
            self.trajectory_pub.publish(self.trajectory_desired)
            self.end_effector_pub.publish(self.end_effector)
            self.robot_joint1_pub.publish(self.joint1)
            self.robot_joint2_pub.publish(self.joint2)
            self.robot_joint3_pub.publish(self.joint3)
        except CvBridgeError as e:
            print(e)


# call the class
def main(args):
    ic = image_converter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


# run the code if the node is called
if __name__ == '__main__':
    print("Image processing v2 attempt w file save")
    main(sys.argv)
