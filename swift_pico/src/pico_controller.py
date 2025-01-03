#!/usr/bin/env python3

'''
This python file runs a ROS 2-node of name pico_control which holds the position of Swift Pico Drone on the given dummy.
This node publishes and subscribes to the following topics:

        PUBLICATIONS            SUBSCRIPTIONS
        /drone_command           /whycon/poses
        /pid_error               /throttle_pid
                                /pitch_pid
                                /roll_pid
'''

# Importing the required libraries

from swift_msgs.msg import SwiftMsgs
from geometry_msgs.msg import PoseArray
from pid_msg.msg import PIDTune, PIDError
import rclpy
from rclpy.node import Node
import numpy as np


class Swift_Pico(Node):
    def __init__(self):
        super().__init__('pico_controller')  # initializing ros node with name pico_controller
        
        # This corresponds to your current position of the drone.
        # This value must be updated each time in your whycon callback [x, y, z]
        self.drone_position = np.array([0.0, 0.0, 0.0])

        # whycon marker at the position of the dummy given in the scene.
        self.setpoint = np.array([2.0, 2.0, 19.0])  # [x_setpoint, y_setpoint, z_setpoint]
        self.cmd = SwiftMsgs()  # Declaring a cmd of message type swift_msgs and initializing values
        self.cmd.rc_roll = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_throttle = 1500

        # Initial setting of Kp, Kd, and Ki for [roll, pitch, throttle].
        self.Kp = np.array([8.5, 8.5, 12.0])
        self.Ki = np.array([0.01, 0.01, 0.09])
        self.Kd = np.array([260.0, 260.0, 445.0])

        self.scale_kp = np.array([0.1, 0.1, 0.1])       # Scale factors
        self.scale_ki = np.array([0.0001, 0.0001, 0.0001])
        self.scale_kd = np.array([0.1, 0.1, 0.1])

        self.prev_error = np.array([0.0, 0.0, 0.0])     # Previous errors in each axis
        self.error_sum = np.array([0.0, 0.0, 0.0])      # Sum of errors in each axis
        self.max_values = np.array([2000, 2000, 2000])  # Maximum and minimum values in each axis
        self.min_values = np.array([1000, 1000, 1000])
        self.sample_time = 0.060                        # in seconds
        self.count = 0
        self.flag = 0
        self.drone_position_error = self.drone_position - self.setpoint
        # Publishing /drone_command, /pid_error
        self.command_pub = self.create_publisher(SwiftMsgs, '/drone_command', 10)
        self.pid_error_pub = self.create_publisher(PIDError, '/pid_error', 10)

        # Subscribing to /whycon/poses, /throttle_pid, /pitch_pid, roll_pid
        self.create_subscription(PoseArray, '/whycon/poses', self.whycon_callback, 1)
        self.create_subscription(PIDTune, "/throttle_pid", self.throttle_tune_pid, 1)
        self.create_subscription(PIDTune, '/pitch_pid', self.pitch_tune_pid, 1)
        self.create_subscription(PIDTune, '/roll_pid', self.roll_tune_pid, 1)

        self.arm()  # ARMING THE DRONE

        self.timer = self.create_timer(self.sample_time, self.pid)  # Creating a timer to run the pid function periodically

    def disarm(self): 
        self.cmd.rc_roll = 1000
        self.cmd.rc_yaw = 1000
        self.cmd.rc_pitch = 1000
        self.cmd.rc_throttle = 1000
        self.cmd.rc_aux4 = 1000
        self.command_pub.publish(self.cmd)  # Publishing /drone_command

    def arm(self):
        self.disarm()
        self.cmd.rc_roll = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_throttle = 1500
        self.cmd.rc_aux4 = 2000
        self.command_pub.publish(self.cmd)  # Publishing /drone_command

    # Whycon callback function
    # The function gets executed each time when /whycon node publishes /whycon/poses
    def whycon_callback(self, msg):
        self.drone_position[0] = msg.poses[0].position.x 
        self.drone_position[1] = msg.poses[0].position.y
        self.drone_position[2] = msg.poses[0].position.z	

    # Callback function for /throttle_pid
    # This function gets executed each time when /drone_pid_tuner publishes /throttle_pid
    def throttle_tune_pid(self, throttle):
        self.Kp[2] = throttle.kp * self.scale_kp[2]  # Example, you can change the ratio/fraction value accordingly
        self.Ki[2] = throttle.ki * self.scale_ki[2]
        self.Kd[2] = throttle.kd * self.scale_kd[2]

    # Callback function for /pitch_pid
    # This function gets executed each time when /drone_pid_tuner publishes /pitch_pid
    def pitch_tune_pid(self, pitch):
        self.Kp[1] = pitch.kp * self.scale_kp[1]  
        self.Ki[1] = pitch.ki * self.scale_ki[1]                                                    
        self.Kd[1] = pitch.kd * self.scale_kd[1]

    # Callback function for /roll_pid
    # This function gets executed each time when /drone_pid_tuner publishes /roll_pid
    def roll_tune_pid(self, roll):
        self.Kp[0] = roll.kp * self.scale_kp[0]  
        self.Ki[0] = roll.ki * self.scale_ki[0]
        self.Kd[0] = roll.kd * self.scale_kd[0]

    def pid(self):
        self.error = self.drone_position - self.setpoint
        self.error_difference = self.error - self.prev_error
        self.error_sum += self.error
        self.controlled_error = self.Kp * self.error + self.Kd * self.error_difference + self.Ki * self.error_sum
        [self.out_roll, self.out_pitch, self.out_throttle] = self.controlled_error
        self.values = np.array([-self.out_roll, self.out_pitch, self.out_throttle]) + 1500
        for i in range(3):                          
            if self.values[i] >= self.max_values[i]: 
                self.values[i] = self.max_values[i]         #clipping the output to maximum
            elif self.values[i] <= self.min_values[i]: 
                self.values[i] = self.min_values[i]         #clipping the output to minimum
        
        self.cmd.rc_roll = int(self.values[0])              #typecasting PID outputs to integer values
        self.cmd.rc_pitch = int(self.values[1])
        self.cmd.rc_throttle = int(self.values[2]) 
        self.command_pub.publish(self.cmd)                  #Publishing roll, pitch and throttle values
        
        self.prev_error = self.error                        #Updating previous error
        self.drone_position_error = self.drone_position - self.setpoint
        pid_error = PIDError()
        pid_error.roll_error = self.error[0]
        pid_error.pitch_error = self.error[1]
        pid_error.throttle_error = self.error[2]
        self.pid_error_pub.publish(pid_error)               #Publishing roll, pitch and throttle error
        if self.flag<67:
            self.error_sum = [0.0,0.0,0.0]
            self.flag+=1
        else:
            self.count = 0
           
       
def main(args=None):
    rclpy.init(args=args)
    swift_pico = Swift_Pico()
    rclpy.spin(swift_pico)
    swift_pico.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
