import json
import math
import datetime


class RobotPathExporter:
    def __init__(self, nested_paths, start_position, product_shape="irregular_polygon", scale=1.0, bounding_box=None,ellipse=None):
        """
        Initialize the exporter with the given nested paths, starting position, and metadata.
        
        :param nested_paths: The raw nested paths. For example:
                             [
                               [((271, 377), (375, 446)), ((376, 446), (463, 389))],
                               [((317, 290), (335, 335)), ((339, 335), (357, 289))]
                             ]
        :param start_position: The starting position of the robot (x, y).
        :param product_shape: A string describing the shape of the product.
        :param scale: A scale factor (e.g., mm per pixel).
        :param bounding_box: Optional. A dictionary with 'min' and 'max' keys representing the bounding box.
                             For example: {"min": [250, 300], "max": [600, 500]}
        """
        self.nested_paths = nested_paths
        self.start_position = start_position
        self.product_shape = product_shape
        self.scale = scale
        self.bounding_box = bounding_box  # Pre-calculated bounding box, if available.
        # Unpack the nested paths into a flat list of (start, end) tuples.
        self.ellipse=ellipse
        self.paths = self.unpack_paths(self.nested_paths)

    def unpack_paths(self, nested_paths):
        """
        Flattens the nested paths.
        
        If nested_paths is structured as a list of groups, where each group is a list of paths,
        this method returns a single flat list of (start, end) tuples.
        
        :param nested_paths: The nested list of paths.
        :return: A flat list of (start, end) tuples.
        """
        if isinstance(nested_paths[0], list):
            return [path for group in nested_paths for path in group]
        else:
            return nested_paths

    def compute_bounding_box(self):
        """
        Computes the bounding box for all paths if no bounding_box was provided.
        
        :return: A dictionary with 'min' and 'max' keys representing the bounding box.
        """
        xs = []
        ys = []
        for start, end in self.paths:
            xs.extend([start[0], end[0]])
            ys.extend([start[1], end[1]])
        return {"min": [min(xs), min(ys)], "max": [max(xs), max(ys)]}

    def compute_center(self, bbox):
        """
        Computes the center (arithmetic mean) of a bounding box.
        
        :param bbox: A dictionary with 'min' and 'max' keys.
        :return: A list [center_x, center_y].
        """
        center_x = (bbox["min"][0] + bbox["max"][0]) / 2.0
        center_y = (bbox["min"][1] + bbox["max"][1]) / 2.0
        return [center_x, center_y]

    def compute_paths_data(self):
        """
        Computes parameters for each path and its segments while maintaining the grouping.

        :return: A list of dictionaries, each representing a path with its segments.
        """
        paths_data = []

        for path_id, path in enumerate(self.nested_paths):  # Keep paths grouped
            path_segments = []
            previous_angle = None
            total_length = 0.0  # Initialize path length

            for segment_id, (line_type, points) in enumerate(path):
                # Convert numpy types to Python int
                points = [[int(p[0]), int(p[1])] for p in points]

                if line_type == "line":
                    start, end = points
                    dx = end[0] - start[0]
                    dy = end[1] - start[1]
                    length = math.hypot(dx, dy)
                    angle = math.degrees(math.atan2(dy, dx))

                elif line_type == "circ":
                    start, via, end = points
                    # Approximate arc length using control points
                    chord_length = math.hypot(end[0] - start[0], end[1] - start[1])
                    midpoint = [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2]
                    control_dist = math.hypot(via[0] - midpoint[0], via[1] - midpoint[1])
                    length = chord_length + 2 * control_dist  # Rough estimate
                    angle = math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))
                elif line_type=="point":
                    length=0
                    angle=0
                else:
                    raise ValueError(f"Unknown segment type: {line_type}")

                total_length += length  # Accumulate segment length

                if previous_angle is None:
                    angle_diff = 180
                else:
                    angle_diff = abs(angle - previous_angle)
                    if angle_diff > 180:
                        angle_diff = 360 - angle_diff

                base_resolution = 4.0  # mm
                interpolation_resolution = base_resolution if angle_diff < 10 else base_resolution + (angle_diff / 20)
                transition_flag = angle_diff > 15.0

                segment_data = {
                    "segment_id": segment_id,
                    "type": line_type,
                    "points": points,
                    "length": round(length, 2),
                    "angle": round(angle, 2),
                    "angle_difference": round(angle_diff, 2),
                    "interpolation_resolution": round(interpolation_resolution, 2),
                    "transition_flag": transition_flag
                }
                path_segments.append(segment_data)
                previous_angle = angle

            # Add the entire path with its segments and total length
            paths_data.append({
                "path_id": path_id,
                "total_length": round(total_length, 2),  # Include total path length
                "segments": path_segments
            })

        return paths_data

    def create_export_data(self):
        """
        Prepares the full data (header and paths) for export.
        
        :return: A dictionary representing the data to be exported.
        """
        # Use the provided bounding_box if available; otherwise, compute it.
        bbox = self.bounding_box if self.bounding_box is not None else self.compute_bounding_box()
        center_bbox = self.compute_center(bbox)
        center_of_gravity = center_bbox  # You might change this if a different calculation is needed.
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        header_data = {
            "version": "1.0",
            "shape": self.product_shape,
            "bounding_box": bbox,
            "center_of_bounding_box": center_bbox,
            "center_of_gravity": center_of_gravity,
            "ellipse parameters": self.ellipse,
            "units": "pixel",
            "scale": self.scale,
            "description": "Product outline with decoration paths.",
            "Date": f"Exported at {current_time}",
            "start_position": list(self.start_position)
        }

        paths_data = self.compute_paths_data()
        return {"header": header_data, "paths": paths_data}

    def convert_ordered_paths_to_json_structure(self, ordered_paths):
        """
        Converts the ordered paths (a nested list structure) into a JSON-compatible structure.
        
        :param ordered_paths: A list of paths, where each path is a list of segments.
                              Each segment is a tuple: (segment_type, points_tuple)
                              For example:
                              [
                                  [
                                      ('line', ((457, 473), (435, 476))),
                                      ('circ', ((419, 447), (420, 440), (424, 433)))
                                  ],
                                  [
                                      ('line', ((380, 472), (378, 407)))
                                  ]
                              ]
        :return: A list of dictionaries representing the paths.
        """
        paths_json = []
        
        for path_index, path in enumerate(ordered_paths):
            segments_json = []
            for segment in path:
                seg_type = segment[0]
                raw_points = segment[1]
                
                # Convert each point into a list of standard Python integers.
                points = []
                for point in raw_points:
                    # Ensure conversion from numpy types (if present) to int.
                    x = int(point[0])
                    y = int(point[1])
                    points.append([x, y])
                
                segments_json.append({
                    "type": seg_type,
                    "points": points
                })
                
            paths_json.append({
                "id": path_index,
                "segments": segments_json
            })
        
        return paths_json

    def convert_json_to_grouped_paths(self, json_structure):
        """
        Converts a JSON structure (like the one produced by convert_ordered_paths_to_json_structure)
        back into the original grouped list format.
        
        :param json_structure: A list of dictionaries representing paths, each with an "id" and a "segments" key.
                               Each segment has a "type" and "points" (a list of [x, y] pairs).
        :return: A grouped list of paths where each path is a list of segments in the form:
                 (segment_type, points_tuple)
                 For example:
                 [
                    [('line', ((457, 473), (435, 476))),
                     ('circ', ((419, 447), (420, 440), (424, 433)))],
                    [('line', ((380, 472), (378, 407)))]
                 ]
        """
        grouped_paths = []
        for path in json_structure:
            segments = []
            for seg in path.get("segments", []):
                seg_type = seg.get("type")
                # Convert each point (list) back to a tuple
                points = tuple(tuple(point) for point in seg.get("points", []))
                segments.append((seg_type, points))
            grouped_paths.append(segments)
        return grouped_paths,json_structure

    def export_to_json(self, filename):
        """
        Exports the computed header and paths data to a JSON file.
        
        :param filename: The file path for the output JSON.
        """
        data = self.create_export_data()
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Exported robot paths to {filename}")

# --- Example Usage ---

if __name__ == "__main__":
    # Example nested paths (grouped as lists of segments).
    nested_paths = [
        [
            ('line', ((457, 473), (435, 476))),
            ('line', ((433, 475), (427, 473))),
            ('line', ((425, 471), (419, 450))),
            ('circ', ((419, 447), (420, 440), (424, 433))),
            ('line', ((426, 431), (446, 427))),
            ('circ', ((448, 428), (456, 437), (457, 447))),
            ('line', ((457, 448), (419, 448)))
        ],
        [
            ('line', ((380, 472), (378, 407))),
            ('line', ((381, 407), (403, 408)))
        ],
        [
            ('line', ((366, 294), (407, 294)))
        ],
        [
            ('line', ((365, 264), (366, 328)))
        ]
    ]
    start_position = (250, 300)
    bounding_box = {"min": [250, 264], "max": [457, 476]}

    # Create an exporter instance.
    exporter = RobotPathExporter(nested_paths, start_position, product_shape="irregular_polygon", scale=1.0, bounding_box=bounding_box)
    
    # Export header and paths data to a JSON file.
    exporter.export_to_json("robot_paths.json")
    
    # Convert the ordered paths (using our function) to a JSON structure.
    json_structure = exporter.convert_ordered_paths_to_json_structure(nested_paths)
    print("JSON structure for paths:")
    print(json.dumps(json_structure, indent=4))
    
    # Convert the JSON structure back to the original grouped paths.
    grouped_paths_restored = exporter.convert_json_to_grouped_paths(json_structure)
    print("Restored grouped paths:")
    print(grouped_paths_restored)

