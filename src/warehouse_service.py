#!/usr/bin/env python3
import heapq
import rclpy
import cv2
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
from waypoint_navigation.srv import GetPathNew
from waypoint_navigation.msg import Pathway

ROW = 1000
COL = 1000
grid = np.asarray(cv2.imread('2D_bit_map.png', cv2.IMREAD_GRAYSCALE))
i = -1

class Cell:
    def __init__(self):
        self.f = float('inf')
        self.g = float('inf')
        self.h = 0
        self.parent_i = 0
        self.parent_j = 0
        self.waypoints = []

class Path(Node):

    def __init__(self):
        super().__init__('path_planning_service')
        self.subscription = self.create_subscription(
            Int32MultiArray,
            'package_loc',
            self.get_waypoint_callback,
            10)
        self.srv = self.create_service(GetPathNew, 'path_new', self.waypoint_callback)
        self.subscription
        self.path = []
        self.waypoints = [(500, 500)] 
        
    def get_waypoint_callback(self, msg):
        global i
        i = i+1 
        self.waypoints.append((msg.data[0], msg.data[1]))
         
    def waypoint_callback(self, request, response):
        response.waypoints.data = []
        if request.get_path == True:
            response.waypoints.data.append(self.waypoints[0][0])
            response.waypoints.data.append(self.waypoints[0][1])
            response.waypoints.data.append(self.waypoints[1][0])
            response.waypoints.data.append(self.waypoints[1][1])
                
            self.get_logger().info("Incoming request for Waypoints") 
            self.get_logger().info(f"Waypoints: {self.waypoints[0]}, {self.waypoints[1]}")
            self.path = a_star_search(self.waypoints[i], self.waypoints[i+1])   
            if self.path:
                for points in self.path_1:
                    path_point = Pathway()
                    path_point.x = points[0]
                    path_point.y = points[1]
                    response.path1.append(path_point)
                self.get_logger().info("Path output")
            else:
                self.get_logger().info("No path found")
            return response
        else:
            self.get_logger().info("Request rejected")
            
def is_valid(row, col):
    return (row >= 0) and (row < ROW) and (col >= 0) and (col < COL)

def is_unblocked(grid, row, col):
    return grid[row][col] == 255


def is_destination(row, col, dest):
    return row == dest[0] and col == dest[1]

# Calculate the heuristic value of a cell (Euclidean distance to destination)

def calculate_h_value(row, col, dest):
    return ((row - dest[0]) ** 2 + (col - dest[1]) ** 2) ** 0.5

# Trace the path from source to destination
def trace_path(cell_details, dest):
    path = []
    row = dest[0]
    col = dest[1]

    # Trace the path from destination to source using parent cells
    while not (cell_details[row][col].parent_i == row and cell_details[row][col].parent_j == col):
        path.append((row, col))
        temp_row = cell_details[row][col].parent_i
        temp_col = cell_details[row][col].parent_j 
        row = temp_row
        col = temp_col

    # Add the source cell to the path
    path.append((row, col))
    # Reverse the path to get the path from source to destination
    path.reverse()
    return path

# Implement the A* search algorithm
def a_star_search(src, dest):
    # Initialize the closed list (visited cells)
    closed_list = [[False for _ in range(COL)] for _ in range(ROW)]
    # Initialize the details of each cell
    cell_details = [[Cell() for _ in range(COL)] for _ in range(ROW)]

    # Initialize the start cell details
    i = src[0]
    j = src[1]
    cell_details[i][j].f = 0
    cell_details[i][j].g = 0
    cell_details[i][j].h = 0
    cell_details[i][j].parent_i = i
    cell_details[i][j].parent_j = j

    open_list = []
    heapq.heappush(open_list, (0.0, i, j))
    
    while len(open_list) > 0:
        # Pop the cell with the smallest f value from the open list
        p = heapq.heappop(open_list)

        # Mark the cell as visited
        i = p[1]
        j = p[2]
        closed_list[i][j] = True
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0),
                      (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for dir in directions:
            new_i = i + dir[0]
            new_j = j + dir[1]
            
            # If the successor is valid, unblocked, and not visited
            if is_valid(new_i, new_j) and is_unblocked(grid, new_i, new_j) and not closed_list[new_i][new_j]:
                # If the successor is the destination
                if is_destination(new_i, new_j, dest):
                    # Set the parent of the destination cell
                    cell_details[new_i][new_j].parent_i = i
                    cell_details[new_i][new_j].parent_j = j
                    # Trace and print the path from source to destination
                    path = trace_path(cell_details, dest)
                    return path
                else:
                    # Calculate the new f, g, and h values
                    g_new = cell_details[i][j].g + 1.0
                    h_new = calculate_h_value(new_i, new_j, dest)
                    f_new = g_new + h_new
                    # If the cell is not in the open list or the new f value is smaller
                    if cell_details[new_i][new_j].f == float('inf') or cell_details[new_i][new_j].f > f_new:
                        # Add the cell to the open list
                        heapq.heappush(open_list, (f_new, new_i, new_j))
                        # Update the cell details
                        cell_details[new_i][new_j].f = f_new
                        cell_details[new_i][new_j].g = g_new
                        cell_details[new_i][new_j].h = h_new
                        cell_details[new_i][new_j].parent_i = i
                        cell_details[new_i][new_j].parent_j = j

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


