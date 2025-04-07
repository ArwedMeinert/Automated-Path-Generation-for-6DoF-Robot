import numpy as np
import cv2
import functions
import math
import Logger

class Optimisation:
    def __init__ (self,robot_path,margin=1,log_obj=None):
        self.path=robot_path
        self.margin=margin
        self.log_obj=log_obj
        self.sorted_Path=[]

    def find_neighbors_within_margin(self, path_segment, margin, free_endpoint=None):
        """Find neighboring path segments within the margin, optionally using only a given free endpoint."""
        neighbors = []
        
        # Extract points based on segment type
        if path_segment[0] == 'line':
            start, end = path_segment[1]
        elif path_segment[0] == 'circ':
            start, via, end = path_segment[1]
        elif path_segment[0]=='point':
            return
        
        # Determine the relevant reference point
        reference_points = (start, end) if free_endpoint is None else (free_endpoint,)  # If free_endpoint is given, only consider that one

        for other_path_segment in self.path:
            if other_path_segment == path_segment:
                continue  # Skip itself

            if other_path_segment[0] == 'line':
                other_start, other_end = other_path_segment[1]
            elif other_path_segment[0] == 'circ':
                other_start, other_via, other_end = other_path_segment[1]
            elif other_path_segment[0] == 'point':
                other_start, other_end = other_path_segment[1]


            # Check if the free endpoint (or both endpoints if none given) is near the neighbor's endpoints
            for ref in reference_points:
                if (self.distance(ref, other_start) < margin or 
                    self.distance(ref, other_end) < margin):
                    neighbors.append(other_path_segment)
                    break  # Stop checking once it's a valid neighbor

        return neighbors



    def calculate_direction(self,segment):
        """Calculate the direction angle (radians) of a path segment."""
        path_type, points = segment
        if path_type == 'line':
            (x1, y1), (x2, y2) = points
            return np.arctan2(y2 - y1, x2 - x1)
        elif path_type == 'circ':
            (x1, y1), (vx, vy), (x2, y2) = points
            return np.arctan2(vy - y1, vx - x1)  # Approximate arc direction
        return None

    def choose_best_neighbor(self, segment, neighbors):
        """
        Selects the neighbor with the most similar direction to the given segment.
        Accounts for the fact that direction is not important (segments can be reversed).
        """
        seg_dir = self.calculate_direction(segment)  # Get the direction of the current segment
        
        def best_direction_match(neighbor):
            # Calculate original direction
            neighbor_dir = self.calculate_direction(neighbor)
            diff_original = abs(np.arctan2(np.sin(seg_dir - neighbor_dir), np.cos(seg_dir - neighbor_dir)))  # Angle difference in range [-π, π]

            # Calculate reversed direction
            reversed_neighbor = self.reverse_segment(neighbor)
            reversed_dir = self.calculate_direction(reversed_neighbor)
            diff_reversed = abs(np.arctan2(np.sin(seg_dir - reversed_dir), np.cos(seg_dir - reversed_dir)))

            # Return the smallest difference
            return min(diff_original, diff_reversed)
        
        # Find the neighbor with the smallest direction difference (either normal or reversed)
        best_neighbor = min(neighbors, key=best_direction_match, default=None)
        return best_neighbor

    def reverse_segment(self, segment):
        """
        Reverses a segment's start and end points to compare direction correctly.
        Handles both line and arc (circ) segments.
        """
        path_type, points = segment
        if path_type == 'line':
            return ('line', (points[1], points[0]))  # Swap start and end
        elif path_type == 'circ':
            return ('circ', (points[2], points[1], points[0]))  # Swap start and end, keep control point
        else:
            raise ValueError(f"Unknown segment type: {path_type}")



    @Logger.log_execution_time("log_obj")
    def sort_paths(self, margin=2, debug=0, image=None):
        """
        Sort paths so that they form continuous moves.
        
        For each segment:
        - Identify if it can serve as a starting segment (neighbors exist on only one side).
        - Starting from one valid start segment, follow the neighbor chain,
        choosing the best neighbor as the chain expands (and re-searching neighbors).
        - After finishing one chain, update the list of remaining segments and start a new chain.
        - Finally, add any segments that weren't connected.
        
        :param margin: maximum distance to consider endpoints as neighbors.
        :param debug: 0 for no debug, 1 for limited debug, 2 for maximal debug
        :param image: if provided, an image on which to overlay debug information.
        :return: a list of segments in the order they are connected.
        """
        sorted_paths = []   # Final ordered list of segments.
        visited = set()     # To ensure each segment is used only once.
        grouped_paths = []  # Each chain as a separate list.
        image_overlay = image.copy() if image is not None else None

        # Continue until every segment has been processed.
        while len(visited) < len(self.path):
            current_path = []
            candidate_start = None
            candidate_free_endpoint = None

            # Look for a candidate start segment among unvisited segments.
            for seg in self.path:
                if seg in visited:
                    continue

                # Find neighbors using a tighter margin.
                neighbors = self.find_neighbors_within_margin(seg, margin)
                side_counts = self.count_neighbors_on_sides(seg, margin)
                # Choose segments that have neighbors on only one side.
                if neighbors and ((side_counts[0] == 0 and side_counts[1] > 0) or (side_counts[1] == 0 and side_counts[0] > 0)):
                    # Orient segment so that its free endpoint is at the start.
                    if side_counts[0] == 0:
                        candidate_free_endpoint = seg[1][-1]
                        candidate_start = seg
                    else:
                        candidate_free_endpoint = seg[1][0]
                        # Reverse the segment.
                        if seg[0] == 'line':
                            candidate_start = ('line', (seg[1][-1], seg[1][0]))
                        elif seg[0] == 'circ':
                            candidate_start = ('circ', (seg[1][-1], seg[1][1], seg[1][0]))
                        else:
                            candidate_start = seg
                        # Update the segment in self.path.
                        self.path[self.path.index(seg)] = candidate_start
                    break

            # If no valid start segment was found, pick the first unvisited segment.
            if candidate_start is None:
                for seg in self.path:
                    if seg not in visited:
                        candidate_start = seg
                        candidate_free_endpoint = self.extract_free_endpoint(seg)
                        break

            # Start a new chain with the chosen segment.
            sorted_paths.append(candidate_start)
            visited.add(candidate_start)
            current_segment = candidate_start
            current_path.append(current_segment)

            if debug:
                print(f"Starting new chain with: {candidate_start}")
                if image_overlay is not None:
                    image_overlay = functions.overlay_robot_path(
                        image_overlay, [candidate_start], color=(255, 0, 255), line_thickness=2
                    )
                    cv2.imshow("Debug Start Segment", image_overlay)
                    cv2.waitKey(0)
                    image_overlay = image.copy()

            # Chain expansion: look for neighbors based on the current free endpoint.
            free_endpoint = candidate_free_endpoint
            while True:
                # Find all neighbors to the current segment using the provided margin.
                candidates = self.find_neighbors_within_margin(current_segment, margin, free_endpoint)
                if candidates:
                    # Only consider segments not yet visited.
                    candidates = [n for n in candidates if n not in visited]
                if not candidates:
                    break  # No more neighbors; chain is complete.

                best_candidate = self.choose_best_neighbor(current_segment, candidates)
                if best_candidate is None:
                    break

                # Orient the candidate so its free endpoint aligns with the chain.
                oriented_candidate, new_free_endpoint = self.orient_segment_to_endpoint(best_candidate, free_endpoint, margin)
                if oriented_candidate is None:
                    visited.add(best_candidate)
                    continue  # Skip if candidate cannot be properly oriented.

                sorted_paths.append(oriented_candidate)
                visited.add(best_candidate)
                current_segment = oriented_candidate
                free_endpoint = new_free_endpoint
                current_path.append(current_segment)

                if debug:
                    print(f"Added segment: {oriented_candidate}")
                    if image_overlay is not None:
                        image_overlay = functions.overlay_robot_path(
                            image_overlay, [oriented_candidate], color=(0, 255, 255), line_thickness=2
                        )
                        cv2.imshow("Debug Chain Expansion", image_overlay)
                        cv2.waitKey(0)
                        image_overlay = image.copy()

            grouped_paths.append(current_path)

        # Finally, if any segments were missed, add them as isolated paths.
        for seg in self.path:
            if seg not in visited:
                sorted_paths.append(seg)
                visited.add(seg)
                grouped_paths.append([seg])
                if debug:
                    print(f"Adding isolated segment: {seg}")
                    if image_overlay is not None:
                        image_overlay = functions.overlay_robot_path(
                            image_overlay, [seg], color=(200, 200, 100), line_thickness=2
                        )
                        cv2.imshow("Debug Sorting", image_overlay)
                        cv2.waitKey(0)
        sorted_paths,grouped_paths=self.filter_short_paths(sorted_paths, grouped_paths, min_length=8)
        self.sorted_Path = sorted_paths
        return sorted_paths, [grouped_paths]

    def filter_short_paths(self,sorted_paths, grouped_paths, min_length):
        """
        Filters out paths that consist of only one segment (two endpoints),
        are below a specified length, and are not of type "point".
        
        :param sorted_paths: Flat list of all segments (in order).
        :param grouped_paths: List of lists, where each inner list is a chain (path).
        :param min_length: Minimum length required for a two-point path to be kept.
        :return: (filtered_sorted_paths, filtered_grouped_paths)
        """
        filtered_grouped_paths = []
        for path in grouped_paths:
            # Check if this path qualifies for deletion:
            # It must be a single segment (only two points) and not of type "point".
            if len(path) == 1:
                seg = path[0]
                line_type, points = seg
                if line_type != "point":
                    # Calculate the segment's length.
                    if line_type == "line":
                        start, end = points
                        dx = end[0] - start[0]
                        dy = end[1] - start[1]
                        length = math.hypot(dx, dy)
                    elif line_type == "circ":
                        start, via, end = points
                        dx = end[0] - start[0]
                        dy = end[1] - start[1]
                        length = math.hypot(dx, dy)
                    else:
                        length = 0
                    # If the calculated length is below the threshold, skip this path.
                    if length < min_length:
                        continue
            # If the path does not match the deletion criteria, keep it.
            filtered_grouped_paths.append(path)
        
        # Rebuild the flat sorted_paths by flattening the filtered grouped paths.
        filtered_sorted_paths = [segment for chain in filtered_grouped_paths for segment in chain]
        
        return filtered_sorted_paths, filtered_grouped_paths

    def extract_free_endpoint(self, segment):
        """ Determines the free endpoint of a segment (line or arc). """
        path_type, points = segment
        if path_type == 'line':
            return points[1]  # End point of the line
        elif path_type == 'circ':
            return points[2]  # End point of the arc
        elif path_type == 'point':
            return points[1]
        else:
            raise ValueError(f"Unknown segment type: {path_type}")
    

    def orient_segment_to_endpoint(self, segment, active_endpoint, margin):
        """
        Given a candidate segment and the current active endpoint,
        check which endpoint of the candidate is near active_endpoint.
        If needed, reverse the segment so that the connecting endpoint comes first.
        
        Returns a tuple (oriented_segment, new_free_endpoint) if a connection is possible,
        or (None, None) if no endpoint is within the margin.
        """
        seg_type, points = segment
        if seg_type == 'line':
            start, end = points
            if self.distance(active_endpoint, start) < self.distance(active_endpoint, end):
                return segment, end
            else :
                reversed_segment = (seg_type, (end, start))
                return reversed_segment, start
        elif seg_type == 'circ':
            start, via, end = points
            if self.distance(active_endpoint, start) < self.distance(active_endpoint, end):
                return segment, end
            else:
                reversed_segment = (seg_type, (end, via, start))
                return reversed_segment, start
        else:
            print("Unknown segment type")
            return (None, None)


    def count_neighbors_on_sides(self, path_segment, margin=2):
        """
        Counts the number of neighbors on the left and right sides of the path_segment.
        Uses a discrete search in a square area around each endpoint.

        :param path_segment: Tuple (segment_type, points), where:
                            - 'line': points = ((x1, y1), (x2, y2))
                            - 'circ': points = ((x1, y1), (vx, vy), (x2, y2))
        :param neighbors: List of segments in the same format.
        :param margin: The square margin used to detect neighbors.
        :return: (left_count, right_count)
        """
        # Extract endpoints
        path_type, points = path_segment
        if path_type == 'line':
            start, end = points
        elif path_type == 'circ':
            start, _, end = points
        elif path_type == 'point':
            start, end = points
        else:
            raise ValueError("Unknown path type")

        left_count, right_count = 0, 0
        for other_path in self.path:

            path_type, points = other_path
            if other_path==path_segment:
                continue

            if path_type == 'line':
                other_start, other_end = points
            elif path_type == 'circ':
                other_start, _, other_end = points
            elif path_type == 'point':
                other_start,other_end = points

            if (self.distance(end, other_start) < margin or  self.distance(end,other_end)<margin):
                right_count+=1
            
            if (self.distance(start, other_end) < margin or self.distance(start,other_start)<margin):
                left_count+=1

        return left_count, right_count
    def unpack_paths(self,nested_paths):
        # Flatten the outermost path lists (remove the first level of nesting)
        paths= [segment for path in nested_paths for segment in path]
        return [segment for path in paths for segment in path]
    
    



    def distance(self,p1, p2):
        """Calculate Euclidean distance between two points."""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)