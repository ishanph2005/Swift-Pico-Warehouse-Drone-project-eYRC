#!/usr/bin/env python3
import heapq
import rclpy
import cv2
import numpy as np
import random
import math
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
from waypoint_navigation.srv import GetPath
from waypoint_navigation.msg import Pathway

ROW = 1000
COL = 1000
grid = np.asarray(cv2.imread('2D_bit_map.png', cv2.IMREAD_GRAYSCALE))

class Cell:
    def __init__(self):
        self.f = float('inf')
        self.g = float('inf')
        self.h = 0
        self.parent_i = 0
        self.parent_j = 0
        self.waypoints = []
        
class RRTNode:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent

class Path(Node):

    def __init__(self):
        super().__init__('path_planning_service')
        self.subscription = self.create_subscription(
            Int32MultiArray,
            'random_points',
            self.get_waypoint_callback,
            10)
        self.srv = self.create_service(GetPath, 'path', self.waypoint_callback)
        self.path = []
        self.waypoints = []
        self.path_1 = []  # Initialize paths
        self.path_2 = []  

    def get_waypoint_callback(self, msg):
        # Hardcoded initial point for demonstration
        self.waypoints.append((500, 500)) 
        self.waypoints.append((msg.data[0], msg.data[1]))
        self.waypoints.append((msg.data[2], msg.data[3]))

    def waypoint_callback(self, request, response):
        response.waypoints.data = []

        if request.get_path:
            # Append waypoints to the response
            for wp in self.waypoints:
                response.waypoints.data.append(wp[0])
                response.waypoints.data.append(wp[1])

            self.get_logger().info("Incoming request for Waypoints") 
            self.get_logger().info(f"Waypoints: {self.waypoints}")

            # RRT* Path Planning
            self.path_1 = rrt_star(grid, self.waypoints[0], self.waypoints[1])
            self.path_2 = rrt_star(grid, self.waypoints[1], self.waypoints[2])

            # Add the computed paths to the response
            if self.path_1:
                for points in self.path_1:
                    path_point = Pathway()
                    path_point.x = points[0]
                    path_point.y = points[1]
                    response.path1.append(path_point)
                self.get_logger().info("Path 1 generated successfully")
            else:
                self.get_logger().info("No Path 1 found")

            if self.path_2:
                for points in self.path_2:
                    path_point = Pathway()
                    path_point.x = points[0]
                    path_point.y = points[1]
                    response.path2.append(path_point)
                self.get_logger().info("Path 2 generated successfully")
            else:
                self.get_logger().info("No Path 2 found")
        else:
            self.get_logger().info("Request rejected")

        return response

# Helper Functions
def is_valid(row, col):
    return (row >= 0) and (row < ROW) and (col >= 0) and (col < COL)

def is_unblocked(grid, row, col):
    return grid[row][col] == 255

def is_destination(row, col, dest):
    return row == dest[0] and col == dest[1]

def calculate_h_value(row, col, dest):
    return ((row - dest[0]) ** 2 + (col - dest[1]) ** 2) ** 0.5

def trace_path(cell_details, dest):
    path = []
    row, col = dest

    while not (cell_details[row][col].parent_i == row and cell_details[row][col].parent_j == col):
        path.append((row, col))
        temp_row = cell_details[row][col].parent_i
        temp_col = cell_details[row][col].parent_j 
        row = temp_row
        col = temp_col

    path.append((row, col))
    path.reverse()
    return path

# RRT* algorithm
def rrt_star(grid, start, goal, max_iter=1000, step_size=10, radius=15):
    start_node = RRTNode(start[0], start[1])
    goal_node = RRTNode(goal[0], goal[1])
    nodes = [start_node]

    def distance(node1, node2):
        return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

    def is_in_grid(x, y):
        return 0 <= x < ROW and 0 <= y < COL

    def is_unblocked_rrt(x, y):
        return is_in_grid(x, y) and grid[int(x)][int(y)] == 255

    def nearest_node(nodes, random_point):
        return min(nodes, key=lambda node: distance(node, random_point))

    def path_from_start(node):
        path = []
        while node:
            path.append((int(node.x), int(node.y)))
            node = node.parent
        path.reverse()
        return path

    for _ in range(max_iter):
        rand_x = random.randint(0, ROW - 1)
        rand_y = random.randint(0, COL - 1)
        rand_point = RRTNode(rand_x, rand_y)

        nearest = nearest_node(nodes, rand_point)
        theta = math.atan2(rand_y - nearest.y, rand_x - nearest.x)
        new_x = nearest.x + step_size * math.cos(theta)
        new_y = nearest.y + step_size * math.sin(theta)

        if not is_unblocked_rrt(new_x, new_y):
            continue

        new_node = RRTNode(new_x, new_y, nearest)
        nodes.append(new_node)

        # Check for goal proximity
        if distance(new_node, goal_node) < step_size:
            goal_node.parent = new_node
            nodes.append(goal_node)
            return path_from_start(goal_node)

        # Rewiring
        for other_node in nodes:
            if other_node != new_node and distance(new_node, other_node) < radius:
                potential_cost = distance(new_node, start_node) + distance(new_node, other_node)
                if potential_cost < distance(other_node, start_node):
                    other_node.parent = new_node

    return None  # No path found                

def main():
    rclpy.init()
    path = Path() 

    try:
        rclpy.spin(path)
    except KeyboardInterrupt:
        path.get_logger().info('KeyboardInterrupt, shutting down.\n')
    finally:
        path.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
