from html import escape as html_escape
import re
from svgpathtools import CubicBezier
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from sympy import Point,Line
import math
import json

def display_paths(image, paths_json, thickness=2, debug=False):
    """
    Draws all paths from the JSON data onto the given image in different colors.
    
    Each path is assumed to be a dictionary with an 'id' and a list of 'segments'.
    Each segment is a dictionary with:
      - 'type': either 'line' or 'circ' or 'point'
      - 'points': a list of points; for lines, two points; for circular segments, three points.
    
    Additionally, before drawing a segment, if the start point of the current segment is 
    within one pixel (in both x and y) of the endpoint of the previous segment, a warning is printed.
    
    :param image: The input image (numpy array) on which the paths will be drawn.
    :param paths_json: List of path dictionaries as loaded from the JSON file.
    :param thickness: Thickness of the drawn segments.
    :param debug: If True, prints debug information.
    """
    # Make a copy of the image so we don't modify the original.
    img_copy = image.copy()
    
    # Pre-define a palette of colors or generate random colors.
    colors = [
        (255, 0, 0),     # Blue
        (0, 255, 0),     # Green
        (0, 0, 255),     # Red
        (255, 255, 0),   # Cyan
        (255, 0, 255),   # Magenta
        (0, 255, 255),   # Yellow
        (128, 0, 0),     # Maroon
        (0, 128, 0),     # Dark Green
        (0, 0, 128)      # Navy
    ]
    cv2.imshow('image', img_copy)
    cv2.waitKey(0)
    # Iterate over each path in the JSON list.
    for idx, path in enumerate(paths_json):
        # Choose a color for this path.
        color = colors[idx % len(colors)]
        segments = path.get('segments', [])
        
        # Initialize previous endpoint as None for each new path.
        prev_endpoint = None
        
        # Draw each segment.
        for seg in segments:
            seg_type = seg.get('type')
            points = seg.get('points', [])
            
            # Determine the start point (if available) based on segment type.
            current_start = None
            current_end = None
            
            if seg_type == 'line' and len(points) >= 2:
                current_start = (int(points[0][0]), int(points[0][1]))
                current_end   = (int(points[1][0]), int(points[1][1]))
            elif seg_type == 'circ' and len(points) >= 3:
                current_start = (int(points[0][0]), int(points[0][1]))
                current_end   = (int(points[2][0]), int(points[2][1]))
            elif seg_type == 'point' and len(points) >= 1:
                current_start = (int(points[0][0]), int(points[0][1]))
                current_end   = current_start  # Only one point exists.
            
            # Check if the current segment's start is too close to the previous segment's end.
            if prev_endpoint is not None and current_start is not None:
                if abs(current_start[0] - prev_endpoint[0]) > 1 or abs(current_start[1] - prev_endpoint[1]) > 1:
                    print(f"Warning: Endpoint {prev_endpoint} of previous segment is in the neighborhood of start {current_start} of the next segment.")
            
            # Now draw the segment based on its type.
            if seg_type == 'line':
                if len(points) >= 2:
                    p1 = current_start
                    p2 = current_end
                    cv2.line(img_copy, p1, p2, color, thickness)
                    if debug:
                        cv2.imshow('image', img_copy)
                        cv2.waitKey(100)
                        print(f"Path {path.get('id')}: Drew line from {p1} to {p2}")
                else:
                    if debug:
                        print(f"Path {path.get('id')}: Not enough points for line segment.")
            
            elif seg_type == 'circ':
                if len(points) >= 3:
                    p1 = current_start
                    p2 = (int(points[1][0]), int(points[1][1]))
                    p3 = current_end
                    draw_circle_segment(img_copy, p1, p2, p3, color=color, thickness=thickness, debug=debug)
                    if debug:
                        print(f"Path {path.get('id')}: Drew circle segment with points {p1}, {p2}, {p3}")
                else:
                    if debug:
                        print(f"Path {path.get('id')}: Not enough points for circle segment.")
            
            elif seg_type == 'point':
                if len(points) >= 1:
                    p1 = current_start
                    cv2.circle(img_copy, p1, thickness, color, thickness)
                    if debug:
                        print(f"Path {path.get('id')}: Drew point at {p1}")
                else:
                    if debug:
                        print(f"Path {path.get('id')}: Not enough points for point segment.")
            else:
                if debug:
                    print(f"Path {path.get('id')}: Unknown segment type '{seg_type}'.")
            
            # Update the previous endpoint for the next segment check.
            if current_end is not None:
                prev_endpoint = current_end
        
        # Optionally, label the path using its id near the first segment's starting point.
        if debug:
            if segments:
                first_seg = segments[0]
                if first_seg.get('type') == 'line' and len(first_seg.get('points', [])) >= 1:
                    pt = first_seg['points'][0]
                elif first_seg.get('type') == 'circ' and len(first_seg.get('points', [])) >= 1:
                    pt = first_seg['points'][0]
                else:
                    pt = (10, 10)  # fallback position
                cv2.putText(img_copy, str(path.get('id')),
                            (int(pt[0]), int(pt[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Display the final image.
    cv2.imshow("Paths", img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def overlay_robot_path(image, robot_path, line_thickness=1,color=(0,0,255),debug=False):
    """
    Overlay the robot path (linear and arc segments) onto the image.
    :param image: The image to overlay on.
    :param robot_path: List of waypoints and arcs [('line', (start, end)), ('circ', (start, via, end))].
    :param line_thickness: Thickness for the line segments.
    :param arc_thickness: Thickness for the arc segments.
    :return: The image with waypoints and path overlaid.
    """

    overlay_image = image.copy()
    
    for waypoint in robot_path:
        if waypoint[0] == 'line':
            # For line segments, draw a line between the start and end points
            start, end = waypoint[1]
            cv2.line(overlay_image, start, end, color, line_thickness)  # Red line for line segments
        elif waypoint[0] == 'point':
            # For point waypoints, draw a small circle at the point
            start, end = waypoint[1]
            cv2.circle(overlay_image, start, 2, color, -1)
        
        elif waypoint[0] == 'circ':
            # For circular arcs, get the start, via, and end points
            start, via, end = waypoint[1]
            draw_circle_segment(overlay_image,start,via,end,color,line_thickness,debug)
    return overlay_image

def draw_circle_segment(image, p1, p2, p3, color=(255, 0, 0), thickness=2, debug=False):
    """
    Draw a circle segment between three points in OpenCV.

    Args:
    - image: The OpenCV image to draw on.
    - p1, p2, p3: Tuple coordinates of the start, via, and end points (x, y).
    - color: The color of the segment (default: blue).
    - thickness: The thickness of the segment (default: 2).
    """
    # Fit a circle through the three points
    circle_center, radius = fit_circle_through_points(p1, p2, p3)

    if circle_center is None or radius is None:
        print("The points are collinear or invalid for circle fitting.")
        return

    # Calculate angles for the points relative to the circle center
    cx, cy = circle_center
    start_angle = math.degrees(math.atan2(p1[1] - cy, p1[0] - cx))
    via_angle = math.degrees(math.atan2(p2[1] - cy, p2[0] - cx))

    end_angle = math.degrees(math.atan2(p3[1] - cy, p3[0] - cx))
    start_angle = (start_angle +360) % 360
    via_angle = (via_angle + 360) % 360
    end_angle = (end_angle + 360) % 360
    end_angle,via_angle,start_angle = adjust_endpoint_direction(start_angle, via_angle, end_angle)

    if debug:            #    B    G   R
        cv2.circle(image, p1, 2, (255, 255, 0), 1)  # Start
        cv2.circle(image, p2, 2, (0, 255, 255), 1)  # Via
        cv2.circle(image, p3, 2, (255, 0, 255), 1)  # End
        print(f"Start angle: {start_angle:.2f}, Via angle: {via_angle:.2f}, End angle: {end_angle:.2f}")
        cv2.imshow('image', image)
        cv2.waitKey(100)

    # Function to find the shortest angle difference
    def angle_difference(a1, a2):
        diff = (a2 - a1) % 360
        if diff > 180:
            diff -= 360
        return diff

    # Function to check the orientation (clockwise or counterclockwise)
    def cross_product(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

    # Check if the points form a clockwise or counterclockwise arc
    orientation = cross_product(p1, p2, p3)

    if abs(angle_difference(start_angle, end_angle)) > 180:
        # If the arc is larger than 180 degrees, split the arc into two segments
        intermediate_angle1 = (start_angle + via_angle) / 2
        intermediate_x1 = cx + radius * math.cos(math.radians(intermediate_angle1))
        intermediate_y1 = cy + radius * math.sin(math.radians(intermediate_angle1))
        intermediate_point1 = (int(intermediate_x1), int(intermediate_y1))

        intermediate_angle2 = (via_angle + end_angle) / 2
        intermediate_x2 = cx + radius * math.cos(math.radians(intermediate_angle2))
        intermediate_y2 = cy + radius * math.sin(math.radians(intermediate_angle2))
        intermediate_point2 = (int(intermediate_x2), int(intermediate_y2))

        # Recursively draw the two shorter arcs
        draw_circle_segment(image, p1, intermediate_point1, p2, (255, 255, 0), thickness, debug)
        draw_circle_segment(image, p2, intermediate_point2, p3, (0, 255, 255), thickness, debug)
    else:
        # If the points are in counterclockwise order, reverse the direction
        

        # Draw the single arc
        cv2.ellipse(
            image,
            (int(cx), int(cy)),
            (int(radius), int(radius)),
            0,  # Rotation angle of the ellipse
            start_angle,
            end_angle,
            color,
            thickness
        )
        if debug:
            cv2.imshow('image', image)
            cv2.waitKey(100)

    return image


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
    ##print(f"start: {start_angle} via: {via_angle} end: {end_angle}")
    
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

def unpack_paths(nested_paths):
    # Flatten the outermost path lists (remove the first level of nesting)
    return [segment for path in nested_paths for segment in path]

def is_between(start_angle, via_angle, end_angle):
    # Normalize the angles to [0, 360)
    start_angle %= 360
    via_angle %= 360
    end_angle %= 360

    # Determine if the via point is between start and end angles
    if start_angle < end_angle:
        return start_angle < via_angle < end_angle
    else:
        # Handle the wraparound case where angles cross 0 degrees
        return via_angle > start_angle or via_angle < end_angle
    

def open_json(json_file):
    try:
        with open(json_file, "r") as file:
            return json.load(file)  # Load JSON file into dictionary
    except FileNotFoundError:
        print(f"Error: The file {json_file} was not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {json_file}.")
        return []

def convert_json_to_grouped_paths(JSONfile):
    """
    Converts a JSON structure (like the one produced by convert_ordered_paths_to_json_structure)
    back into the original grouped list format.
    
    :param JSONfile: The path to the JSON file containing the paths data.
    :return: A grouped list of paths where each path is a list of segments in the form:
             (segment_type, points_tuple)
             Example:
             [
                [('line', ((457, 473), (435, 476))),
                 ('circ', ((419, 447), (420, 440), (424, 433)))],
                [('line', ((380, 472), (378, 407)))]
             ]
    """
    try:
        with open(JSONfile, "r") as file:
            json_data= json.load(file)  # Load JSON file into dictionary
    except FileNotFoundError:
        print(f"Error: The file {JSONfile} was not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {JSONfile}.")
        return []

    # Extract the "paths" key, which contains the relevant data
    paths_list = json_data.get("paths", [])

    grouped_paths = []
    for path in paths_list:
        segments = []
        for seg in path.get("segments", []):
            seg_type = seg.get("type")
            points = tuple(tuple(point) for point in seg.get("points", []))
            segments.append((seg_type, points))
        grouped_paths.append(segments)

    return grouped_paths,json_data





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








## Vectorgraphics
def extract_svg_text_info(svg_text):
    # Parse the SVG text element
    # Extract the <text> element string
    start = svg_text.find('<text')
    end = svg_text.find('</text>')
    text_element = svg_text[start:end+7]
    
    # Extract attributes and text content
    if text_element:
        start_text = text_element.find('>')
        text_content = text_element[start_text+1:-7]
        
        fill_color = None
        font_family = None
        font_size = None
        
        if 'fill="' in text_element:
            fill_color_start = text_element.find('fill="') + len('fill="')
            fill_color_end = text_element.find('"', fill_color_start)
            fill_color = text_element[fill_color_start:fill_color_end]
        
        if 'font-family="' in text_element:
            font_family_start = text_element.find('font-family="') + len('font-family="')
            font_family_end = text_element.find('"', font_family_start)
            font_family = text_element[font_family_start:font_family_end]
        
        if 'font-size="' in text_element:
            font_size_start = text_element.find('font-size="') + len('font-size="')
            font_size_end = text_element.find('"', font_size_start)
            font_size = text_element[font_size_start:font_size_end]
        
        return {
            'text': text_content,
            'fill_color': fill_color,
            'font_family': font_family,
            'font_size': font_size,
            'text_element_string': text_element
        }
    else:
        return None


def text2svg(
    text: str,
    fill: str = "#fff",
    font: str = "sans-serif",
    size: float = 16,
    baseline: float = 1,
    padding: float = 1,
    ratio: float = 1,  # usually 2 for monospace
) -> str:
    """convert count to svg
    fill -- text colour
    font -- font family
    size -- font size in pixels
    baseline -- baseline offset
    padding -- padding of characters
    ratio -- character ratio
    embedding :
    <img
        id="my-stuff"
        src="..."
        style="display:inline;height:1em;vertical-align:top"
        alt="my stuff :3"
    />
    """

    fill = html_escape(fill)
    font = html_escape(font)

    svg: str = f'<svg xmlns="http://www.w3.org/2000/svg" width="{len(text) + padding * ratio}ch" height="{size}" font-size="{size}">'
    svg += f'<text x="50%" y="{size - baseline}" text-anchor="middle" fill="{fill}" font-family="{font}">{html_escape(text)}</text>'
    svg += "</svg>"

    return svg

def extract_font_from_svg(svg_content):
    font_data=[]
    font_type = re.search(r'<font-face[^>]*>', svg_content)
    if font_type:
        return font_type.group(0)
    return None

def is_non_linear_path(path):
    # Check if the path has more than two points and contains non-linear segments
    if len(path) > 2:
        for segment in path:
            if isinstance(segment, CubicBezier):
                return True
    return False

def is_path_longer_than(path, min_length):
    # Calculate the total length of the path
    total_length = sum(segment.length() for segment in path)
    # Return True if the total length is greater than or equal to min_length, otherwise False
    return total_length >= min_length
