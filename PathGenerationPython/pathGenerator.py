from html import escape as html_escape
import re
from svgpathtools import CubicBezier
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import math
import Logger


class Skeleton_to_path:
    def __init__(self, skeleton=None,margin=2,min_line_length=5,debug=False,max_line_iterations=0,min_connected_points_to_filter=2,log_obj=None):
        '''
        Creates an instance of the path generator. It generates path segments from the skeleton of shapes.
        
        :param skeleton: bitmap of a skeleton image
        :param margin: specifies the margin the line can deviate from the skeleton
        :param min_line_length: specifies the minimal length a line can have. no shorter lines will be generated
        :param debug: if true debug information is shown
        :param max_line_iterations: when a valid line has been found, the start of the line will be changed to try to increase the lenght of the line
        :param :param max_line_iterations: sets the minimum of connected points so that the points are not filtered out after the line createion.
        '''
        self.max_points_for_line_test=max_line_iterations
        self.skeleton = skeleton
        self.debug=debug
        self.margin=margin
        self.min_line_length=min_line_length
        self.min_connected_points_to_filter=min_connected_points_to_filter
        self.log_obj=log_obj
        if self.skeleton is not None:
            self.orederd_waypoints = self.extract_ordered_waypoints_from_skeleton()
            self.robot_path,self.remaining_points=self.generate_robot_path_from_skeleton()
        else:
            self.orederd_waypoints = None
            self.robot_path,self.remaining_points=None,None

        self.min_path_length=float('inf')
    @Logger.log_execution_time("log_obj")
    def extract_ordered_waypoints_from_skeleton(self):
        """
        Extract and order waypoints from the skeleton, ensuring sequential connectivity
        within each contour, without mixing contours.
        :return: List of ordered waypoints for each contour.
        """
        # Find contours of the skeleton
        contours, _ = cv2.findContours(self.skeleton.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours_waypoints = []

        for contour in contours:
            # Flatten the contour and convert points to (x, y) tuples
            ordered_contour = [tuple(point[0]) for point in contour]
            contours_waypoints.append(ordered_contour)  # Append each contour's ordered waypoints as a separate list

        return contours_waypoints

    def is_point_on_line(self, start_point, end_point, test_point, margin=1):
        """
        Check if a point lies within the margin of a line segment.
        :param start_point: Starting point of the line (x, y).
        :param end_point: Ending point of the line (x, y).
        :param test_point: Point to check (x, y).
        :param margin: Allowable margin for the point to be considered on the line.
        :return: True if the point is within the margin, False otherwise.
        """
        start = np.array(start_point)
        end = np.array(end_point)
        test = np.array(test_point)

        line_vec = end - start
        point_vec = test - start
        line_len = np.linalg.norm(line_vec)

        if line_len == 0:
            return np.linalg.norm(point_vec) <= self.margin

        projection = np.dot(point_vec, line_vec) / line_len
        closest_point = start + projection * (line_vec / line_len)

        distance_to_line = np.linalg.norm(test - closest_point)
        return distance_to_line <= self.margin
    
    @Logger.log_execution_time("log_obj")
    def generate_line_segments(self, contours_waypoints, robot_path, min_line_length=5, max_points_to_test=0):
        '''
        Generates line segments from sorted waypoints that are nestled by contours.
        
        :param contours_waypoints: waypoints attached to a contour
        :param robot_path: already existing path. Can be used to shorten the min_line_length parameter. Otherwise an empty list can be passed.
        :param min_line_length: specifies the minimal length a line can have. No shorter lines will be generated.
        :param max_points_to_test: when a valid line has been found, the start of the line will be changed to try to increase the length of the line.
        '''
        leftover_points_by_contour = []
        # Predefine neighborhood offsets as a tuple to avoid recreating on each iteration.
        neighborhood_offsets = (
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),          (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        )
        for contour in contours_waypoints:
            leftover_in_contour = []
            if len(contour) < 2:
                leftover_points_by_contour.append(contour)
                continue

            # Create a set of available points for O(1) membership tests and efficient removals.
            available_points = set(tuple(pt) for pt in contour)

            while available_points:
                # --- Optimize starting point selection ---
                # Instead of looping and checking each point, use a list comprehension to find candidates
                # with exactly one neighbor.
                candidates = [
                    point for point in available_points 
                    if sum(((point[0] + dx, point[1] + dy) in available_points) for dx, dy in neighborhood_offsets) == 1
                ]
                if candidates:
                    start_point = candidates[0]
                else:
                    start_point = next(iter(available_points))
                # -------------------------------------------------

                # Generate a line segment starting from the chosen point using a copy of available_points.
                line_points, line_length = self.generate_line_from_start(start_point, available_points.copy())

                # Evaluate alternatives by testing early segments if enabled.
                best_line_points = line_points
                best_line_length = line_length
                first_line_points = line_points.copy()
                if max_points_to_test > 0:
                    for i, test_point in enumerate(line_points[:max_points_to_test]):
                        test_line_points, test_line_length = self.generate_line_from_start(test_point, available_points.copy())
                        # Update best line if the candidate is significantly longer.
                        if test_line_length > best_line_length + i:
                            best_line_points = test_line_points
                            best_line_length = test_line_length

                # If a valid line segment is found, add it to the robot_path.
                if best_line_length >= min_line_length:
                    start_arr = np.array(best_line_points[0], dtype=int)
                    end_arr = np.array(best_line_points[-1], dtype=int)
                    robot_path.append(('line', (tuple(start_arr), tuple(end_arr))))
                    if best_line_length < self.min_line_length:
                        self.min_line_length = best_line_length

                    # --- Optimize point removal ---
                    # If an alternative best line was selected, remove points preceding the best line's start.
                    if best_line_points != first_line_points:
                        try:
                            idx = first_line_points.index(best_line_points[0])
                        except ValueError:
                            idx = 0
                        for pt in first_line_points[:idx]:
                            available_points.discard(pt)
                            leftover_in_contour.append(pt)
                    # Remove all points used in the best line.
                    for pt in best_line_points:
                        available_points.discard(pt)
                else:
                    # If no valid line is found, remove the starting point.
                    available_points.discard(start_point)
                    leftover_in_contour.append(start_point)
                # -------------------------------------------------
                
                # Debug visualization if enabled.
                if self.debug:
                    self.debug_visualize(best_line_points, available_points)
            leftover_points_by_contour.append(leftover_in_contour)
        return robot_path, leftover_points_by_contour


    def generate_line_from_start(self, start_point, remaining_points_to_check):
        """
        Generate a line segment starting from a given point, ensuring all intermediate points are within margin.
        :param start_point: The starting point of the line.
        :param remaining_points_to_check: The set of points still available.
        :return: Tuple (line_points, line_length).
        """
        line_points = [start_point]
        # Remove the start point from available points.
        remaining_points_to_check.discard(start_point)
        # Precompute a NumPy array of the remaining points for vectorized distance checks.
        remaining_points_arr = np.array(list(remaining_points_to_check))
        current_point = start_point
        neighborhood_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),         (0, 1),
            (1, -1), (1, 0), (1, 1)
        ]

        # Helper function to compute squared Euclidean distance to avoid sqrt overhead.
        def squared_distance(p1, p2):
            dx = p1[0] - p2[0]
            dy = p1[1] - p2[1]
            return dx * dx + dy * dy

        while True:
            # Find valid neighbor points using the set membership of remaining_points_to_check.
            neighbors = [
                (current_point[0] + dx, current_point[1] + dy)
                for dx, dy in neighborhood_offsets
                if (current_point[0] + dx, current_point[1] + dy) in remaining_points_to_check
            ]
            if not neighbors:
                break

            # Select the closest neighbor using the squared distance.
            next_point = min(neighbors, key=lambda p: squared_distance(p, current_point))

            # Retrieve intermediate points between the start and this candidate next point.
            intermediate_points = self.get_intermediate_points(start_point, next_point, step_size=2)
            
            # Validate that every intermediate point is close enough to at least one of the original remaining points.
            valid = True
            for ip in intermediate_points:
                ip_arr = np.array(ip)  # Convert once per intermediate point.
                distances = np.linalg.norm(remaining_points_arr - ip_arr, axis=1)
                if not np.any(distances <= self.margin):
                    valid = False
                    break

            if valid:
                line_points.append(next_point)
                current_point = next_point
                remaining_points_to_check.discard(next_point)
                # Note: We keep remaining_points_arr constant, mirroring the original behavior.
            else:
                break

        # Compute the overall line length as the Euclidean distance between the first and last points.
        start = np.array(line_points[0])
        end = np.array(line_points[-1])
        line_length = np.linalg.norm(end - start)

        return line_points, line_length


    def get_intermediate_points(self, start_point, end_point, step_size=0.5):
        """
        Generate intermediate points between two points.
        :param start_point: Start of the line segment (x, y).
        :param end_point: End of the line segment (x, y).
        :param step_size: Distance between intermediate points.
        :return: List of intermediate points (x, y).
        """
        start = np.array(start_point)
        end = np.array(end_point)
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)
        num_steps = int(line_len // step_size)

        intermediate_points = [
            tuple((start + (t / num_steps) * line_vec).astype(int))
            for t in range(1, num_steps)
        ]
        return intermediate_points

    def debug_visualize(self, line_points, remaining_points):
        # Create a blank image
        img = np.array(self.skeleton, dtype=np.uint8) * 255
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Draw remaining points in red
        for pt in remaining_points:
            cv2.circle(img, pt, 2, (0, 0, 255), -1)

        # Draw line points in blue
        for pt in line_points:
            cv2.circle(img, pt, 2, (255, 0, 0), -1)

        # Draw lines between line points

        # Display the image
        cv2.imshow('Debug Visualization', img)
        cv2.waitKey(200)
        #cv2.destroyAllWindows()
    @Logger.log_execution_time("log_obj")
    def generate_robot_path_from_skeleton(self):
        """
        Generate the robot path from the skeleton, approximating it with multiple linear segments.
        :param margin: The allowed margin (in pixels) for a point to be considered part of a line.
        :param min_line_length: The minimum length for a valid line segment.
        :return: Tuple containing the list of robot path segments and the remaining waypoints grouped by contour.
        """
        contours_waypoints = self.orederd_waypoints
        robot_path = []
        robot_path,leftover_points_by_contour=self.create_small_segments_from_waypoints(contours_waypoints,min_connected_points=1,max_connected_points=3)
        
        if self.debug: print(robot_path)
        robot_path, leftover_points_by_contour = self.generate_line_segments(contours_waypoints, robot_path, min_line_length=self.min_line_length,max_points_to_test=self.max_points_for_line_test)
        #return robot_path, leftover_points_by_contour
        #leftover_points_by_contour=self.filter_leftover_points_by_connectivity_and_adjacency(robot_path.copy(), leftover_points_by_contour, min_connected_points=self.min_connected_points_to_filter)
        robot_path,leftover_points_by_contour=self.create_circular_waypoints_from_line_segments(robot_path, leftover_points_by_contour)
        # Second call with different parameters, using the remaining points from the first call

        #robot_path, leftover_points_by_contour = self.generate_line_segments(leftover_points_by_contour, robot_path, min_line_length=self.min_line_length/2)

        return robot_path, leftover_points_by_contour
    @Logger.log_execution_time("log_obj")
    def create_small_segments_from_waypoints(self, leftover_points_by_contour, min_connected_points=1, max_connected_points=3):
        """
        Process leftover points for each contour to detect small interconnected clusters.
        For each cluster whose size is between min_connected_points and max_connected_points (inclusive),
        generate a small segment represented as a single point.

        Clusters that are smaller than min_connected_points are considered noise and are dropped.
        Clusters that are larger than max_connected_points are considered valid groups of points and
        are passed through for further processing.

        :param leftover_points_by_contour: List of lists of leftover points for each contour.
                                        Each point is assumed to be a tuple, e.g., (x, y).
        :param min_connected_points: Minimum number of points required for a cluster to be considered.
                                    Clusters smaller than this will be dropped.
        :param max_connected_points: Maximum number of points allowed for a cluster to be considered "small".
                                    Clusters with more points are not converted into a small segment.

        :return: A tuple (small_segments, filtered_leftover_points_by_contour)
                - small_segments: A list of single points created from small clusters.
                Each segment is of the form ('point', (x, y)).
                - filtered_leftover_points_by_contour: The leftover points per contour that were not processed
                as small clusters.
        """
        # Define the 8-connected neighborhood offsets.
        neighborhood_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        small_segments = []  # This will hold any generated small segments.
        filtered_leftover_points_by_contour = []  # This will hold the points for clusters that are not converted.
        
        # Process each contour separately.
        for contour_points in leftover_points_by_contour:
            clusters = []   # To store clusters found within this contour.
            visited = set() # To keep track of points that have been clustered.
            
            # Cluster the points using a DFS (flood-fill) approach.
            for point in contour_points:
                if point in visited:
                    continue
                cluster = set()
                stack = [point]
                while stack:
                    p = stack.pop()
                    if p in visited:
                        continue
                    visited.add(p)
                    cluster.add(p)
                    # Check all 8 neighbors.
                    for dx, dy in neighborhood_offsets:
                        neighbor = (p[0] + dx, p[1] + dy)
                        if neighbor in contour_points and neighbor not in visited:
                            stack.append(neighbor)
                clusters.append(cluster)
            
            # Now process each cluster.
            contour_filtered_points = []
            for cluster in clusters:
                cluster_size = len(cluster)
                if cluster_size < min_connected_points:
                    # If the cluster is too small, consider it noise and drop it.
                    continue
                elif cluster_size <= max_connected_points:
                    # The cluster is small enough to be represented as a single point.
                    # Compute the centroid (average position) of the cluster.
                    sum_x = sum(p[0] for p in cluster)
                    sum_y = sum(p[1] for p in cluster)
                    n = cluster_size
                    centroid = (int(sum_x / n), int(sum_y / n))
                    
                    # Append the new point segment.
                    small_segments.append(('point', (tuple(centroid), tuple(centroid))))
                else:
                    # The cluster is too large to be considered a small segment.
                    # Keep all of its points for further processing.
                    contour_filtered_points.extend(cluster)
            
            filtered_leftover_points_by_contour.append(contour_filtered_points)
        
        return small_segments, filtered_leftover_points_by_contour

    @Logger.log_execution_time("log_obj")
    def filter_leftover_points_by_connectivity_and_adjacency(self, robot_path, leftover_points_by_contour, min_connected_points=2):
        """
        Filter the leftover points based on connectivity and adjacency to line segments.
        
        :param robot_path: List of tuples, where each tuple represents a line segment ('line', (start, end)).
        :param leftover_points_by_contour: List of lists of leftover points for each contour.
        :param min_connected_points: Minimum number of connected points for a point to be kept.
        
        :return filtered_leftovers: List of filtered leftover points for each contour.
        """
        # Create a list to store the filtered leftover points
        filtered_leftovers = []
        
        # Extract line segments (start and end points) from robot_path
        line_segments = []
        for waypoint_type, points in robot_path:
            if waypoint_type == 'line':
                start, end = points  # 'line' has two points
                line_segments.append((start, end))
            elif waypoint_type == 'circ':
                start, via, end = points  # 'circ' has three points
                line_segments.append((start, end))  # Use start and end for adjacency checks
        
        # Define neighborhood offsets to check connectivity
        neighborhood_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),         (0, 1),
            (1, -1), (1, 0), (1, 1)
        ]
        
        for leftover_in_contour in leftover_points_by_contour:
            filtered_leftover_in_contour = []
            
            for point in leftover_in_contour:
                # Check if the point is adjacent to either the start or end of any line segment
                is_adjacent_to_line = False
                for line_start, line_end in line_segments:
                    if self.is_point_adjacent_to_line(point, line_start, line_end):
                        is_adjacent_to_line = True
                        break
                
                # If the point is adjacent to a line segment, keep it in the filtered list
                if is_adjacent_to_line:
                    filtered_leftover_in_contour.append(point)
                    continue  # Skip the check for connectivity
                
                # Count connected points in the neighborhood if it's not adjacent to a line
                neighbors = [
                    (point[0] + dx, point[1] + dy)
                    for dx, dy in neighborhood_offsets
                    if (point[0] + dx, point[1] + dy) in leftover_in_contour
                ]
                
                # If the point has fewer than min_connected_points, it gets filtered out
                if len(neighbors) >= min_connected_points:
                    filtered_leftover_in_contour.append(point)

            filtered_leftovers.append(filtered_leftover_in_contour)
        
        return filtered_leftovers


    def is_point_adjacent_to_line(self,point, line_start, line_end):
        """
        Check if a point is adjacent to a line segment.
        
        :param point: The point to check (x, y).
        :param line_start: The start of the line segment (x, y).
        :param line_end: The end of the line segment (x, y).
        
        :return True if the point is adjacent to the line segment, False otherwise.
        """
        x1, y1 = line_start
        x2, y2 = line_end
        xp, yp = point

        # Check if point is near the line segment
        distance_to_start = np.linalg.norm(np.array([xp - x1, yp - y1]))
        distance_to_end = np.linalg.norm(np.array([xp - x2, yp - y2]))

        return distance_to_start <= 1 or distance_to_end <= 1  # Adjust the threshold if needed

    def simulate_robot_path(self, image,path, speed=5, time_interval=0.1,debug=None):
        """
        Simulate the robot's movement along the generated path, adding small circles to the image.
        
        :param image: The image to simulate the robot path on.
        :param robot_path: List of robot path segments as tuples (start_point, end_point) or ('circ', (start, via, end)).
        :param speed: The robot's speed in pixels per time step (default is 5).
        :param time_interval: Time interval in seconds for each movement update (default is 0.1s).
        """
        if debug is None:
            debug=self.debug
        # Create a copy of the image to simulate on
        simulation_image = image.copy()
        circle_radius = 5  # The radius of the circle representing the robot
        
        # Time simulation loop
        for segment in path:
            if debug:print("Processing segment:", segment)
            if segment[0] == 'point':
                # Point segment handling (omitted for brevity; assumed correct)
                start,end = segment[1]
                print(start)
                cv2.circle(simulation_image, (int(start[0]),int(start[1])), circle_radius, (0, 255, 255), -1)
                image = cv2.resize(simulation_image.copy(), (800, 600))
                cv2.imshow("Robot Path Simulation", image)
                cv2.waitKey(int(time_interval * 1000))
            if segment[0] == 'line':
                # Linear segment handling (omitted for brevity; assumed correct)
                start_point = segment[1][0]
                end_point = segment[1][1]
                current_position = np.array(start_point, dtype=np.float64)
                segment_distance = np.linalg.norm(np.array(end_point) - np.array(start_point))
                if segment_distance == 0:
                    continue
                num_steps = int(segment_distance // speed)
                if num_steps == 0:
                    num_steps = 1
                direction = (np.array(end_point) - np.array(start_point)) / segment_distance
                
                for step in range(num_steps):
                    current_position += direction * speed
                    cv2.circle(simulation_image, tuple(current_position.astype(int)), circle_radius, (0, 255, 0), -1)
                    image = cv2.resize(simulation_image.copy(), (800, 600))
                    cv2.imshow("Robot Path Simulation", image)
                    cv2.waitKey(int(time_interval * 1000))
                current_position = np.array(end_point)
                cv2.circle(simulation_image, tuple(current_position.astype(int)), circle_radius, (0, 255, 0), -1)
            
            elif segment[0] == 'circ':
                # Arc (circular) segment
                try:
                    start_point, via_point, end_point = segment[1]
                except Exception as e:
                    print("Error unpacking circular segment:", segment[1])
                    continue

                # Debug prints: show the three points
                if debug:
                    print("Circular segment points:")
                    print("Start:", start_point)
                    print("Via:", via_point)
                    print("End:", end_point)

                # Compute circle parameters using your helper function.
                center, radius = fit_circle_through_points(start_point, via_point, end_point)
                if center is None:
                    print("fit_circle_through_points returned None (points may be collinear). Skipping this segment.")
                    continue  # Skip this segment instead of returning immediately

                if debug:
                    print("Fitted circle center:", center, "radius:", radius)

                cx, cy = center
                # Compute angles (in degrees)
                start_angle = math.degrees(math.atan2(start_point[1] - cy, start_point[0] - cx))
                via_angle = math.degrees(math.atan2(via_point[1] - cy, via_point[0] - cx))
                end_angle = math.degrees(math.atan2(end_point[1] - cy, end_point[0] - cx))
                start_angle = (start_angle + 360) % 360
                via_angle = (via_angle + 360) % 360
                end_angle = (end_angle + 360) % 360

                if debug:
                    print("Initial angles (deg): start =", start_angle, "via =", via_angle, "end =", end_angle)

                # Adjust angles (assuming adjust_endpoint_direction is a helper you defined)
                end_angle, via_angle, start_angle = adjust_endpoint_direction(start_angle, via_angle, end_angle)
                if debug:
                    print("Adjusted angles (deg): start =", start_angle, "via =", via_angle, "end =", end_angle)

                # Compute the arc angle and length.
                arc_angle = end_angle - start_angle
                arc_length = 2 * math.pi * radius * (abs(arc_angle) / 360)
                arc_steps = int(arc_length / speed)
                if arc_steps == 0:
                    arc_steps = 1

                if debug:
                    print("Arc angle (deg):", arc_angle, "Arc length:", arc_length, "Arc steps:", arc_steps)

                for step in range(arc_steps):
                    current_angle = start_angle + (step / arc_steps) * arc_angle
                    current_x = center[0] + radius * np.cos(np.radians(current_angle))
                    current_y = center[1] + radius * np.sin(np.radians(current_angle))
                    cv2.circle(simulation_image, (int(current_x), int(current_y)), circle_radius, (0, 255, 0), -1)
                    image = cv2.resize(simulation_image.copy(), (800, 600))
                    cv2.imshow("Robot Path Simulation", image)
                    cv2.waitKey(int(time_interval * 1000))
                cv2.circle(simulation_image, (int(end_point[0]), int(end_point[1])), circle_radius, (0, 255, 0), -1)

            image = cv2.resize(simulation_image, (800, 600))
            cv2.imshow("Robot Path Simulation", image)
            cv2.waitKey(int(time_interval * 1000))
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return simulation_image


    @Logger.log_execution_time("log_obj")
    def create_circular_waypoints_from_line_segments(self, robot_path, remaining_waypoints):
        """
        Create circular waypoints from line segments by connecting adjacent segments.

        :param robot_path: List of tuples, where each tuple represents a line segment ('line', (start, end)).
        :param remaining_waypoints: List of contours, each containing lists of leftover waypoints.

        :return robot_path: Updated list with new circular waypoints ('circ', (start, via, end)).
        """
        leftover_points_by_contour = []
        for contour in remaining_waypoints:
            leftover_points=[]
            # Group interconnected waypoints within the contour
            connected_points = self.group_interconnected_waypoint_lines(robot_path, contour, min_connected_points=2)

            # Iterate over each group of connected points
            for connections in connected_points:
                if len(connections) < 2:
                    leftover_points.extend(connections)
                    continue  # Skip groups that are too small to form a circular waypoint

                # Define the start, via, and end points for the circular waypoint
                start = connections[0]
                end = connections[-1]
                via = connections[int(len(connections) / 2)]
                if (self.are_points_collinear(start, via, end) and math.sqrt((start[0]-end[0])**2+(start[1]-end[1])**2)>self.margin) or end==via:
                    robot_path.append(('line', (start, end)))
                else:
                    # Add the circular waypoint to the robot path
                    robot_path.append(('circ', (start, via, end)))
                    if self.min_line_length>self.arc_length(start,via,end):
                        self.min_line_length=self.arc_length(start,via,end)
                
            leftover_points_by_contour.append(leftover_points)
        return robot_path,leftover_points_by_contour



    def arc_length(self,start, via, end):
        """
        Approximates the length of the arc described by three points using circle geometry.
        
        :param start: Tuple (x, y) representing the start point of the arc.
        :param via: Tuple (x, y) representing a midpoint on the arc.
        :param end: Tuple (x, y) representing the end point of the arc.
        :return: Approximate arc length.
        """
        # Convert points to numpy arrays
        A, B, C = np.array(start), np.array(via), np.array(end)
        
        # Compute chord lengths
        a = np.linalg.norm(C - B)  # Distance between via and end
        b = np.linalg.norm(A - C)  # Distance between start and end
        c = np.linalg.norm(A - B)  # Distance between start and via
        
        # Semi-perimeter
        s = (a + b + c) / 2

        # Compute triangle area using Heron's formula
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))

        # Compute circumradius R = (abc) / (4 * area)
        if area == 0:
            return np.linalg.norm(A - C)  # If collinear, return straight-line distance
        R = (a * b * c) / (4 * area)

        # Compute the central angle using the law of cosines
        theta = 2 * np.arcsin(b / (2 * R))  # Angle subtended at the circle's center

        # Compute arc length: L = R * θ
        arc_len = R * theta
        
        return arc_len



    def are_points_collinear(self,start, via, end):
        """
        Check if three points are collinear within a given margin.
        :param start: The first point (x1, y1).
        :param via: The second point (x2, y2).
        :param end: The third point (x3, y3).
        :param margin: The allowable margin of error.
        :return: True if points are collinear within the margin, False otherwise.
        """
        x1, y1 = start
        x2, y2 = via
        x3, y3 = end
        
        # Calculate the area of the triangle formed by the points
        area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        
        # If the area is zero (or within the margin), the points are collinear
        return area <= self.margin

    def group_interconnected_waypoint_lines(self, robot_path, leftover_in_contour, min_connected_points=2):
        """
        Group leftover waypoints into interconnected waypoint lines, starting or ending with
        line segment waypoints when applicable.

        :param robot_path: List of tuples, where each tuple represents a line segment ('line', (start, end)).
        :param leftover_in_contour: List of leftover waypoints for the current contour.
        :param min_connected_points: Minimum number of connected points to form a group.

        :return grouped_waypoints: List of grouped waypoint lines for the contour.
        """
        # Extract line segments (start and end points) from robot_path
        line_segments = []
        for waypoint_type, points in robot_path:
            if waypoint_type == 'line':
                start, end = points  # 'line' has two points
                line_segments.append((start, end))
            elif waypoint_type == 'circ':
                start, via, end = points  # 'circ' has three points
                line_segments.append((start, end))  # Use start and end for adjacency checks

        # Define neighborhood offsets to check adjacency
        neighborhood_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),         (0, 1),
            (1, -1), (1, 0), (1, 1)
        ]

        grouped_waypoints = []
        visited = set()  # Track visited points to avoid duplication

        for point in leftover_in_contour:
            if point in visited:
                continue

            # Start a new group and perform BFS to find all connected points
            group = []
            queue = [point]

            while queue:
                current_point = queue.pop(0)
                if current_point in visited:
                    continue

                # Add current point to the group and mark as visited
                group.append(current_point)
                visited.add(current_point)

                # Find neighbors in the contour
                neighbors = [
                    (current_point[0] + dx, current_point[1] + dy)
                    for dx, dy in neighborhood_offsets
                    if (current_point[0] + dx, current_point[1] + dy) in leftover_in_contour
                ]
                queue.extend([neighbor for neighbor in neighbors if neighbor not in visited])

            # Order the points in the group to form a connected line
            ordered_group = self.order_points_in_line(group)

            # Check if the group should start or end with a line segment waypoint
            start_point = ordered_group[0]
            end_point = ordered_group[-1]

            if self.is_adjacent_to_line_segment(start_point, line_segments):
                start_point = start_point  # Use the adjacent point
            if self.is_adjacent_to_line_segment(end_point, line_segments):
                end_point = end_point  # Use the adjacent point

            # Ensure minimum connected points are met
            if len(ordered_group) >= min_connected_points:
                grouped_waypoints.append(ordered_group)

        return grouped_waypoints
    
    def order_points_in_line(self,group):
        """
        Order the points in the group from one end to the other by projecting them onto
        the best-fit line determined via PCA. This method avoids issues with greedy nearest‐neighbor
        ordering when the starting point is not at an extreme.

        Args:
            group (list of tuple): List of points (tuples) to be ordered. For example: [(x1,y1), (x2,y2), ...]

        Returns:
            list of tuple: The points ordered along the best-fit line.
        """
        if len(group) < 2:
            return group  # Nothing to order if one or no point

        # Convert to a NumPy array (shape: N x 2)
        pts = np.array(group, dtype=float)
        
        # Compute the centroid of the points
        centroid = np.mean(pts, axis=0)
        
        # Center the points
        centered_pts = pts - centroid

        # Compute the principal components via SVD (or PCA)
        # The first principal component (direction of maximum variance) is given by Vt[0]
        U, S, Vt = np.linalg.svd(centered_pts, full_matrices=False)
        direction = Vt[0]  # This is a unit vector along the best-fit line

        # Project each point onto the principal direction
        # The projection scalar is computed as the dot product with the direction vector.
        projections = centered_pts.dot(direction)
        
        # Sort the indices of the points by their projection value.
        sorted_indices = np.argsort(projections)
        
        # Order the points accordingly
        ordered_pts = pts[sorted_indices]

        # Convert back to a list of tuples (rounding if desired)
        ordered_group = [tuple(map(int, pt)) for pt in ordered_pts]
        return ordered_group


    def distance(self,p1, p2):
        """Calculate the Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def are_points_adjacent(self, point1, point2, neighborhood_offsets):
        """Check if two points are adjacent based on the neighborhood offsets."""
        for dx, dy in neighborhood_offsets:
            if (point1[0] + dx, point1[1] + dy) == point2:
                return True
        return False


    def is_adjacent_to_line_segment(self, point, line_segments):
        """Check if a point is adjacent to any line segment."""
        neighborhood_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),         (0, 1),
            (1, -1), (1, 0), (1, 1)
        ]

        for line_start, line_end in line_segments:
            if self.are_points_adjacent(point, line_start, neighborhood_offsets) or \
            self.are_points_adjacent(point, line_end, neighborhood_offsets):
                return True
        return False


def fit_circle_through_points(p1, p2, p3):
    """
    Fit a circle through three points.

    Args:
    - p1, p2, p3: Tuple coordinates of three points (x, y).

    Returns:
    - circle_center: The center of the circle (x, y).
    - radius: The radius of the circle.
    """
    # Unpack points
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    
    # Calculate the determinants
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    
    if d == 0:
        return None, None  # Return None for collinear points
    
    # Calculate the coordinates of the center
    ux = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / d
    uy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / d
    
    # Calculate the radius
    r = np.sqrt((ux - x1)**2 + (uy - y1)**2)
    
    if math.isnan(ux) or math.isnan(uy) or math.isnan(r):
        return None, None  # Return None if invalid circle

    return (ux, uy), r


def adjust_endpoint_direction(start_angle, via_angle, end_angle):
    """
    Adjust the endpoint angle to match the direction from start to via point.
    
    Args:
    - start_angle: The starting angle (in degrees).
    - via_angle: The via (intermediate) angle (in degrees).
    - end_angle: The end angle (in degrees).
    
    Returns:
    - Adjusted start angle, via angle, and end angle.
    """
    # Normalize angles to be within the range [0, 360)
    start_angle = (start_angle + 360) % 360
    via_angle = (via_angle + 360) % 360
    end_angle = (end_angle + 360) % 360
    
    # Normalize via angle if needed to handle the wrapping case
    if via_angle - start_angle > 180:
        via_angle -= 360 

    # Adjust the angles based on the relative position
    if start_angle < via_angle < end_angle:
        end_angle, via_angle, start_angle = end_angle, via_angle, start_angle
    elif start_angle > via_angle > end_angle:
        end_angle, via_angle, start_angle = end_angle, via_angle, start_angle
    elif start_angle < via_angle and via_angle > end_angle:
        end_angle += 360
    elif start_angle > via_angle and via_angle < end_angle:
        end_angle -= 360

    # Ensure start_angle is less than end_angle
    if start_angle > end_angle:
        start_angle, end_angle = end_angle, start_angle
    
    # Handle cases where the difference between the start and end angles exceeds 360 degrees
    if end_angle - start_angle > 360:
        start_angle += 360 * 2  # Add 360 twice to wrap it within the 360-degree range

    return start_angle, via_angle, end_angle