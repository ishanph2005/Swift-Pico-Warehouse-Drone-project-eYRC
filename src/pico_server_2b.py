#!/usr/bin/env python3
import math
import numpy as np
from tf_transformations import euler_from_quaternion

import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

# import the action
from waypoint_navigation.action import WaypointPath
from waypoint_navigation.srv import GetWaypoints

# pico control specific libraries
from swift_msgs.msg import SwiftMsgs
from pid_msg.msg import PIDTune, PIDError
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, PoseArray
from waypoint_navigation.msg import CoordinateList

i = 1

class MovingAverageFilter:
    def __init__(self):
        self.window_size = 20
        self.data = []

    def apply(self, new_value):
        # Add the new value to the data list
        self.data.append(new_value)

        # Remove the oldest value if the window size is exceeded
        if len(self.data) > self.window_size:
            self.data.pop(0)

        # Compute and return the average of the current window
        return sum(self.data) / len(self.data)

class WayPointServer(Node):

    def __init__(self):
        super().__init__('path_server')
        print("Node initiated")
        self.path = [[], []]
        self.pid_callback_group = ReentrantCallbackGroup()
        self.action_callback_group = ReentrantCallbackGroup()

        self.time_inside_sphere = 0
        self.max_time_inside_sphere = 0
        self.point_in_sphere_start_time = None
        self.duration = 0

        self.drone_position = np.array([0.0, 0.0, 0.0, 0.0])
        self.setpoint = np.array([0.0, 0.0, 27.0, 0.0]) 
        self.dtime = 0

        self.cmd = SwiftMsgs()
        self.cmd.rc_roll = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_throttle = 1500

        # Kp, Ki and Kd values here
        self.Kp = np.array([8.5, 8.5, 12.0, .0])
        self.Ki = np.array([0.01, 0.01, 0.09, 0.0])
        self.Kd = np.array([260.0, 260.0, 445.0, 0.0])

        # Variables for storing different kinds of errors
        self.prev_error = np.array([0.0, 0.0, 0.0, 0.0])     # Previous errors in each axis
        self.error_sum = np.array([0.0, 0.0, 0.0, 0.0])      # Sum of errors in each axis
        self.max_values = np.array([2000, 2000, 2000, 2000])  # Maximum and minimum values in each axis
        self.min_values = np.array([1000, 1000, 1000, 1000])


        self.pid_error = PIDError()
        self.sample_time = 0.060

        self.command_pub = self.create_publisher(SwiftMsgs, '/drone_command', 10)
        self.pid_error_pub = self.create_publisher(PIDError, '/pid_error', 10)

        self.create_subscription(PoseArray, '/whycon/poses', self.whycon_callback, 1)
        self.create_subscription(PIDTune, "/throttle_pid", self.altitude_set_pid, 1)
        # Add other subscribers here
        self.create_subscription(PIDTune, '/pitch_pid', self.pitch_set_pid, 1)
        self.create_subscription(PIDTune, '/roll_pid', self.roll_set_pid, 1)

        self.create_subscription(Odometry, '/rotors/odometry', self.odometry_callback, 10)

        # Create an action server for the action 'NavToWaypoint'.
        self._action_server = ActionServer(
            self,
            WaypointPath,
            'waypoint_path',
            self.execute_callback,
            callback_group=self.action_callback_group
        )
        self.timer = self.create_timer(self.sample_time, self.pid, callback_group=self.pid_callback_group)
        self.arm()
        
    def disarm(self):
        self.cmd.rc_roll = 1000
        self.cmd.rc_yaw = 1000
        self.cmd.rc_pitch = 1000
        self.cmd.rc_throttle = 1000
        self.cmd.rc_aux4 = 1000
        self.command_pub.publish(self.cmd)

    def arm(self):
        self.disarm()
        self.cmd.rc_roll = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_throttle = 1500
        self.cmd.rc_aux4 = 2000
        self.command_pub.publish(self.cmd)

    def whycon_callback(self, msg):
        self.drone_position[0] = msg.poses[0].position.x
        self.drone_position[1] = msg.poses[0].position.y
        self.drone_position[2] = msg.poses[0].position.z
        self.dtime = msg.header.stamp.sec

    def altitude_set_pid(self, alt):
        self.Kp[2] = alt.kp * 0.1 
        self.Ki[2] = alt.ki * 0.001
        self.Kd[2] = alt.kd * 1.0

    def roll_set_pid(self, roll):
        self.Kp[0] = roll.kp * 0.1 
        self.Ki[0] = roll.ki * 0.001
        self.Kd[0] = roll.kd * 1.0

    def pitch_set_pid(self, pitch):
        self.Kp[1] = pitch.kp * 0.1 
        self.Ki[1] = pitch.ki * 0.001
        self.Kd[1] = pitch.kd * 1.0
        
    def yaw_set_pid(self, yaw):
        self.Kp[3] = yaw.kp * 0.1 
        self.Ki[3] = yaw.ki * 0.001
        self.Kd[3] = yaw.kd * 1.0

    def odometry_callback(self, msg):
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        roll, pitch, yaw = euler_from_quaternion(orientation_list)

        self.roll_deg = math.degrees(roll)
        self.pitch_deg = math.degrees(pitch)
        self.yaw_deg = math.degrees(yaw)
        self.drone_position[3] = self.yaw_deg

    def pid(self):
        sma_filter = MovingAverageFilter()
        noisy_data = self.drone_position - self.setpoint
        self.filtered_data = sma_filter.apply(noisy_data)
        self.error = self.filtered_data
        self.error_difference = self.error - self.prev_error
        self.error_sum += self.error 
        self.controlled_error = self.Kp * self.error + self.Kd * self.error_difference + self.Ki * self.error_sum
        [self.out_roll, self.out_pitch, self.out_throttle] = self.controlled_error[0:3]
        self.values = np.array([-self.out_roll, self.out_pitch, self.out_throttle]) + 1500

        for i in range(3):
            if self.values[i] >= self.max_values[i]:
                self.values[i] = self.max_values[i]  # Clipping the output to maximum
            elif self.values[i] <= self.min_values[i]:
                self.values[i] = self.min_values[i]  # Clipping the output to minimum

        self.cmd.rc_roll = int(self.values[0])  # Typecasting PID outputs to integer values
        self.cmd.rc_pitch = int(self.values[1])
        self.cmd.rc_throttle = int(self.values[2])
        self.prev_error = self.error  # Updating previous error
        self.drone_position_error = self.drone_position - self.setpoint
        pid_error = PIDError()
        pid_error.roll_error = self.error[0]
        pid_error.pitch_error = self.error[1]
        pid_error.throttle_error = self.error[2]
        pid_error.yaw_error = self.error[3]

        self.command_pub.publish(self.cmd)
        self.pid_error_pub.publish(pid_error)
        
    def is_reached_waypoint(self, point):
        return (np.linalg.norm(point[:2] - self.drone_position[:2]) <= 0.04)  
            

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing path...')
        
        # Initialize paths
        self.path[0] = []
        self.path[1] = []

        # Populate path[0] and path[1]
        for points in goal_handle.request.path1:
            self.path[0].append((points.x, points.y, points.z, points.theta))
        for points in goal_handle.request.path2:
            self.path[1].append((points.x, points.y, points.z, points.theta))

        # Check if paths are valid
        if not self.path[0]:
            self.get_logger().error("Path1 is empty.")
            goal_handle.abort()
            return WaypointPath.Result()
        if not self.path[1]:
            self.get_logger().error("Path2 is empty.")
            goal_handle.abort()
            return WaypointPath.Result()

        # Reset internal state
        self.max_time_inside_sphere = 0
        self.point_in_sphere_start_time = None
        self.time_inside_sphere = 0
        self.duration = self.dtime

        feedback_msg = WaypointPath.Feedback()
        self.setpoint = self.path[0][0]

        while True:
            # Populate feedback message
            feedback_msg.current_waypoint.pose.position.x = self.drone_position[0]
            feedback_msg.current_waypoint.pose.position.y = self.drone_position[1]
            feedback_msg.current_waypoint.pose.position.z = self.drone_position[2]
            feedback_msg.current_waypoint.header.stamp.sec = self.max_time_inside_sphere

            global i
            if self.is_reached_waypoint(self.setpoint):
                i += 1
                if i < len(self.path[0]):
                    self.setpoint = self.path[0][i]
                    print(f"WP next: {i}: {self.setpoint}")
                else:
                    print("Completed Path1.")
                    break
            else:
                print(f"Following {i}: {self.setpoint}")
            
            # Final destination check
            if np.sum(self.setpoint == self.path[0][-1]) == 4 or np.sum(self.setpoint == self.path[1][-1]) == 4:
                drone_is_in_sphere = self.is_drone_in_sphere(self.drone_position, 0.6)
                if drone_is_in_sphere:
                    self.time_inside_sphere += self.sample_time
                    if self.time_inside_sphere >= 3:
                        goal_handle.succeed()
                        result = WaypointPath.Result()
                        result.hov_time = self.duration
                        return result

        # Set result for goal completion
        goal_handle.succeed()
        result = WaypointPath.Result()
        result.hov_time = self.duration
        return result



    def is_drone_in_sphere(self, drone_position, radius):
        goal_position = self.destination
        distance = np.linalg.norm(drone_position[:3] - goal_position[:3])
        return distance <= radius

def main(args=None):
    rclpy.init(args=args)

    waypoint_server = WayPointServer()

    executor = MultiThreadedExecutor()
    executor.add_node(waypoint_server)

    try:
        executor.spin()
    finally:
        waypoint_server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

