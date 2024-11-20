#!/usr/bin/env python3
import cv2
import numpy as np
import cv2.aruco as aruco
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image # Image is the message type
from cv_bridge import CvBridge

class Arena(Node):

    def __init__(self):
        super().__init__('image_subscriber')
        self.width = 1000
        self.height = 1000
        self.detected_markers = []
        self.obstacles = 0
        self.subscription = self.create_subscription(
            Image,
            'arena_display/output',
            self.identification,
            10)
        self.subscription  # prevent unused variable warning
        self.br = CvBridge()

    def identification(self, data):

        # Read the image
        self.get_logger().info('Receiving video frame')
    
        # Convert ROS Image message to OpenCV image
        self.frame = self.br.imgmsg_to_cv2(data)
        gray_image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        ###################################
        # Identify the Aruco ID's in the given image
        # define names of each possible ArUco tag OpenCV supports      

        # loop over the types of ArUco dictionaries
       
        # load the ArUCo dictionary, grab the ArUCo parameters, and
        # attempt to detect the markers for the current dictionary
        arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        arucoParams = cv2.aruco.DetectorParameters_create()
        (_, _, self.corners) = cv2.aruco.detectMarkers(gray_image, arucoDict, parameters=arucoParams)
        # if at least one ArUco marker was detected display the ArUco
        # name to our terminal
        ###################################
        # Apply Perspective Transform
        self.ref = np.float32(self.corners[0])

        h, w = self.height, self.width

        dst = np.float32([[w, 0], [w, h], [0, h], [0, 0]])
        matrix = cv2.getPerspectiveTransform(self.ref, dst)
        result = cv2.warpPerspective(self.frame, matrix, (w, h))
        
        transformed_image = result
        ###################################
        # Use the transformed image to find obstacles and their area
        gray = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the original image with a thickness of 10
        cv2.drawContours(transformed_image, contours, -1, (0, 0, 0), 10)

        # Convert the contoured image to grayscale for further processing
        graybitmap = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY)

        # Apply binary thresholding to create a binary bitmap
        _, self.bitmap = cv2.threshold(graybitmap, 150, 255, cv2.THRESH_BINARY)
        
        # Create a mask from the binary image with contours
        mask = np.zeros_like(self.bitmap)
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)  # Fill contours on the mask

        # Dilate the mask to create an inflated mask
        kernel = np.ones((10, 10), np.uint8)
        inflated_mask = cv2.dilate(mask, kernel, iterations=1)
        inflated_image = cv2.bitwise_and(self.bitmap, self.bitmap, mask=inflated_mask)
        inflated_image[460:540, 460:540] = 255
        self.get_logger().info("bitmap image generated")
        cv2.imwrite("2D_bit_map.png", inflated_image)
        ###################################

def main(args=None):
  
    # Initialize the rclpy library
    rclpy.init(args=args)
  
    # Create the node
    image_subscriber = Arena()
  
    # Spin the node so the callback function is called.
    rclpy.spin_once(image_subscriber)
  
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    image_subscriber.destroy_node()
  
    # Shutdown the ROS client library for Python
    rclpy.shutdown()

if __name__ == '__main__':
    main()

