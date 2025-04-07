import numpy as np
import itertools
from python_tsp.heuristics import solve_tsp_local_search
from python_tsp.exact import solve_tsp_brute_force
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_local_search
import functions
import Logger


class GlobalOptimiser:
    def __init__(self,grouped_paths,start_point=(0,0),debug=False,log_obj=None):
                self.grouped_paths=grouped_paths
                self.start_point=start_point
                self.debug=False
                self.log_obj=log_obj
                self.waypoints=self.extract_start_end_points(self.grouped_paths)
                self.dis_matrix,_=self.construct_path_distance_matrix(self.waypoints,self.start_point)
                #print(self.dis_matrix)

    @Logger.log_execution_time("log_obj")
    def tsp_heuristic_solver(self):
         permutation, distance = solve_tsp_local_search(self.dis_matrix)
         ordered_path=self.reorder_paths(self.grouped_paths,permutation)
         return ordered_path,permutation,distance
    
    @Logger.log_execution_time("log_obj")
    def tsp_solver_brute_force(self):
        """Solve TSP using brute force (Held-Karp) - Only feasible for small N"""
        permutation, distance = solve_tsp_brute_force(self.dis_matrix)
        ordered_path=self.reorder_paths(self.grouped_paths,permutation)
        return ordered_path,permutation,distance
    
    @Logger.log_execution_time("log_obj")
    def tsp_solver_dynamic(self):
        """Solve TSP using brute force (Held-Karp) - Only feasible for small N"""
        permutation, distance = solve_tsp_dynamic_programming(self.dis_matrix, maxsize=None)
        ordered_path=self.reorder_paths(self.grouped_paths,permutation)
        return ordered_path,permutation,distance
    
    @Logger.log_execution_time("log_obj")
    def tsp_solver_local(self,startlist=None):
        """Solve TSP using brute force (Held-Karp) - Only feasible for small N"""
        permutation, distance = solve_tsp_local_search(self.dis_matrix, startlist,max_processing_time=4)
        ordered_path=self.reorder_paths(self.grouped_paths,permutation)
        return ordered_path,permutation,distance
    
    @Logger.log_execution_time("log_obj")
    def construct_path_distance_matrix(self, paths, start_position=None,move_to_start=False):
        """
        Constructs a distance matrix where each element [i, j] represents the distance from the endpoint of path i 
        to the startpoint of path j.
        
        If start_position is provided, it is used to compute an initial cost from the robot's starting position
        to each path's start.
        """
        num_paths = len(paths)
        # Initialize a num_paths x num_paths matrix.
        dist_matrix = np.zeros((num_paths, num_paths))
        
        for i, (start_i, end_i) in enumerate(paths):
            for j, (start_j, end_j) in enumerate(paths):
                if i == j:
                    dist_matrix[i, j] = np.inf  # no self-transition
                else:
                    dist_matrix[i, j] = self.distance(end_i, start_j)
        
        # Optionally compute the initial cost from the starting position to each path's start.
        if not move_to_start:
            dist_matrix[:,0]=0
        if start_position is not None:
            initial_costs = [self.distance(start_position, path[0]) for path in paths]
            # You might return these so that your TSP solver can start from start_position.
            return dist_matrix, initial_costs
        else:
            return dist_matrix, None




    def extract_start_end_points(self, grouped_paths):
        """Converts grouped paths into a list of (start, end) tuples."""
        extracted_paths = []
        
        for group in grouped_paths:
            for path in group:
                #print("groupe:", group)
                #print("path:", path)

                # Ensure points are converted to standard Python integers
                start_point = tuple(map(int, path[0][1][0]))  # First segment's start point
                end_point = tuple(map(int, path[-1][1][1]))   # Last segment's end point

                extracted_paths.append((start_point, end_point))
        
        return extracted_paths

    
    def reorder_paths(self, grouped_paths, optimal_order):
        """Reorders the grouped paths based on the optimal TSP solution."""
        grouped_paths=grouped_paths[0]
        # Ensure indices in optimal_order are within range
        if max(optimal_order) >= len(grouped_paths):
            raise IndexError(f"Index {max(optimal_order)} is out of range for grouped_paths of length {len(grouped_paths)}")
        
        # Reorder grouped_paths according to optimal_order
        reordered_paths = [grouped_paths[i] for i in optimal_order]
        
        return reordered_paths






    
    def distance(self,p1, p2):
        """Calculate Euclidean distance between two points."""
        return float(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))