#!/usr/bin/env python3

import time
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from waypoint_navigation.action import WaypointPath
from waypoint_navigation.srv import GetPath
from waypoint_navigation.msg import Pathway
from waypoint_navigation.msg import CoordinateList

class WayPointClient(Node):

    def __init__(self):
        super().__init__('waypoint_client')
        self.full_path1 = []
        self.full_path2 = []
        self.path_index = 0

        # Action client for the action 'NavToWaypoint' with action name 'waypoint_navigation'
        self._action_client = ActionClient(self, WaypointPath, 'waypoint_path')
        
        # Service client for the service 'GetWaypoints' with service name 'waypoints'
        self.cli = self.create_client(GetPath, 'path')

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.get_logger().info("Service started")

        # Request object for GetWaypoints service
        self.request = GetPath.Request()
        
    ### Action client functions

    def send_goal(self):

        # Create a NavToWaypoint goal object
        path_msg = WaypointPath.Goal()
        path_msg.path1 = []
        path_msg.path2 = []
        for point in self.full_path1:
            pathway = CoordinateList()
            pathway.x = point[0]  # Assuming point is a tuple or list with x, y, z coordinates
            pathway.y = point[1]
            pathway.z = point[2]
            pathway.theta = 0.0
            path_msg.path1.append(pathway)  # Append the populated Pathway object to path1
# Create Pathway objects for full_path2 and append them to path2
        for point in self.full_path2:
            pathway = CoordinateList()
            pathway.x = point[0]  # Assuming point is a tuple or list with x, y, z coordinates
            pathway.y = point[1]
            pathway.z = point[2]
            pathway.theta = 0.0
            path_msg.path2.append(pathway)
        print("Goal sent")
        # Waits for the action server to be available
        self._action_client.wait_for_server()
        self.send_goal_future = self._action_client.send_goal_async(path_msg, feedback_callback = self.feedback_callback)   
        self.send_goal_future.add_done_callback(self.goal_response_callback)
        
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info('Result: {0}'.format(result.hov_time))
        self.path_index += 1
        if self.path_index >= len(self.full_path1)-1:
            self.get_logger().info('All waypoints have been reached successfully')  

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        x = feedback.current_waypoint.pose.position.x
        y = feedback.current_waypoint.pose.position.y
        z = feedback.current_waypoint.pose.position.z
        t = feedback.current_waypoint.header.stamp.sec
        self.get_logger().info(f'{x}, {y}, {z}, {t}')

    # Service client functions

    def send_request(self):
        self.request.get_path = True
        self.future = self.cli.call_async(self.request)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
    
    def receive_goals(self):
        response = self.send_request()
        points = Pathway()   
        for points in response.path1:
            
            imgx = points.x
            imgy = points.y
            goal_next = self.pixel_to_whycon(imgx, imgy) 
            self.full_path1.append(goal_next)
        self.get_logger().info('Path1 received by the action client')
        for points in response.path2:
            imgx = points.x
            imgy = points.y
            goal_next = self.pixel_to_whycon(imgx, imgy) 
            self.full_path2.append(goal_next)
        self.get_logger().info('Path2 received by the action client')
        self.send_goal()
            
    def pixel_to_whycon(self, imgx, imgy):
        goal_x= 0.02537*imgx - 12.66
        goal_y= 0.02534*imgy - 12.57
        goal_z= 27.0
        goal = [goal_x, goal_y, goal_z]
        return goal
            
def main(args=None):
    rclpy.init(args=args)

    waypoint_client = WayPointClient()
    waypoint_client.receive_goals()

    try:
        rclpy.spin(waypoint_client)
    except KeyboardInterrupt:
        waypoint_client.get_logger().info('KeyboardInterrupt, shutting down.\n')
    finally:
        waypoint_client.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

