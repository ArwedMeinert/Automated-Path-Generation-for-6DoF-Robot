import cv2
import numpy as np
import functions
import datetime
import json
import os
import cv2.aruco as aruco
import Logger


class AlignShapes:
    def __init__(self,JSON_structure=None,picture=None,AutoMode=False,debug=False,scale=0.3,log_obj=None):
        """
        Initialises the Align shapes class. It should align the shape defined in the
        JSON header with the actual shape of the Object and give back transformation data (rotation, stretching, etc)

        :param JSON_structure: JSON structure with the paths and the header file. The paths should be in respect to pixel coordinates
        :param picture: Picture with the shape that should be aligned with the JSON shape
        :param AutoMode: In auto mode, a camera should be used to take a picture automatically
        """

        if picture==None and AutoMode==False: raise Exception("Either a picture or Auto Mode needs to be passed!")
        self.auto=AutoMode
        if self.auto:
            self.cap=cv2.VideoCapture(0)
        else:
            self.picture=picture
        self.json=JSON_structure
        self.debug=debug
        self.image=None
        self.scale=scale
        self.log_obj=log_obj

        self.marker_size=99 #mm
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.marker_origin=(0,0)
        params = self.load_camera_parameters()  # Defaults to "Camera Config.json"
        if params is None:
            print("No parameter File could be found. Please Calibrate the Camera Parameters")
        else:
            self.camera_params = params
            self.camera_matrix = np.array(self.camera_params["camera_matrix"], dtype=np.float32)
            self.dist_coeffs = np.array(self.camera_params["distortion_coefficients"], dtype=np.float32)

    
    @staticmethod
    def nothing(x):
        pass

    def configure_camera(self):
        """
        Configures the camera by processing an image (live from the camera or from a file)
        and displaying a live preview (or a preview of the static image) with adjustable parameters.
        
        In live mode, frames are captured continuously from the camera.
        In file mode (live==False), a single image (self.picture) is loaded.
        
        The function performs undistortion, ArUco detection, and several image processing steps,
        and provides a debug window with trackbars for tuning parameters.
        
        When 's' is pressed, the current trackbar values are saved to "Camera Config.json".
        Press 'q' to quit.
        """
        # Load existing debug parameters from JSON (or use defaults)
        if os.path.exists("Camera Config.json"):
            with open("Camera Config.json", "r") as f:
                loaded_params = json.load(f)
            self.debug_params = loaded_params
            bk = self.debug_params.get("blur_kernel", 23)
            ab = self.debug_params.get("adaptive_block", 11)
            ac = self.debug_params.get("adaptive_c", 2)
            cmin = self.debug_params.get("canny_min", 50)
            cmax = self.debug_params.get("canny_max", 150)
            min_cont_pixel=self.debug_params.get("min_cont_pixel",50)
            print("Loaded camera config from JSON.")
        else:
            bk, ab, ac, cmin, cmax,min_cont_pixel= 23, 11, 2, 50, 150,50

        # Create a debug window and trackbars
        debug_window = "Debug Controls"
        cv2.namedWindow(debug_window)
        cv2.createTrackbar("Blur Kernel", debug_window, bk, 120, self.nothing)
        cv2.createTrackbar("Adaptive Block", debug_window, ab, 400, self.nothing)
        cv2.createTrackbar("Adaptive C", debug_window, ac, 30, self.nothing)
        cv2.createTrackbar("Canny Min", debug_window, cmin, 255, self.nothing)
        cv2.createTrackbar("Canny Max", debug_window, cmax, 255, self.nothing)
        cv2.createTrackbar("Min Pixel per contour", debug_window, min_cont_pixel, 100, self.nothing)

        # Define a helper to capture an image (live or file)
        if self.auto:
            cap = cv2.VideoCapture(0)  # use default camera index (adjust if needed)
            if not cap.isOpened():
                print("Cannot open camera.")
                return
        else:
            # Load the static image from self.picture
            if not os.path.exists(self.picture):
                print("Picture file not found:", self.picture)
                return
            image = cv2.imread(self.picture)
            if image is None:
                print("Failed to load image from file.")
                return

        while True:
            if self.auto:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame from camera.")
                    break
                orig_image = frame.copy()
            else:
                orig_image = image.copy()

            # Undistort and resize the image
            undistorted_img = cv2.undistort(orig_image, self.camera_matrix, self.dist_coeffs)
            #image_resized = cv2.resize(undistorted_img, (0, 0), fx=self.scale, fy=self.scale)
            
            height, width, _ = undistorted_img.shape
            image_pixels = height * width

            # Convert to grayscale
            gray_image = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)

            # Detect ArUco markers
            corners, ids, _ = cv2.aruco.detectMarkers(gray_image, self.aruco_dict, parameters=self.parameters)
            if corners == ():
                aruco_detected = False
            else:
                aruco_detected = True
            # Create an empty mask
            aruco_mask = np.zeros_like(gray_image)
            if ids is not None and 0 in ids:
                index = np.where(ids == 0)[0][0]
                marker_corners = corners[index].reshape(4, 2).astype(int)
                cv2.fillPoly(aruco_mask, [marker_corners], 255)
                cv2.polylines(undistorted_img, [marker_corners], True, (0, 0, 255), 3)

            # Get current trackbar positions and enforce odd values where needed
            blur_kernel = cv2.getTrackbarPos("Blur Kernel", debug_window)
            adaptive_block = cv2.getTrackbarPos("Adaptive Block", debug_window)
            adaptive_c = cv2.getTrackbarPos("Adaptive C", debug_window)
            canny_min = cv2.getTrackbarPos("Canny Min", debug_window)
            canny_max = cv2.getTrackbarPos("Canny Max", debug_window)
            min_cont_pixel=cv2.getTrackbarPos("Min Pixel per contour", debug_window)
            if blur_kernel % 2 == 0:
                blur_kernel += 1
            if blur_kernel <= 0:
                blur_kernel = 1
            if adaptive_block % 2 == 0:
                adaptive_block += 1

            # Apply processing steps
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray_image)
            blurred = cv2.GaussianBlur(enhanced, (blur_kernel, blur_kernel), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, adaptive_block, adaptive_c)
            kernel = np.ones((2, 2), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            edges = cv2.Canny(thresh, canny_min, canny_max)

            # Find contours in the edge image
            contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
            # Remove any very large contours (likely the whole image)
            while contours_sorted and cv2.contourArea(contours_sorted[0]) > min_cont_pixel/100*image_pixels:
                contours_sorted.pop(0)
            contours_filtered = contours_sorted

            # Exclude contours that fall inside the ArUco marker's expanded bounding box
            margin_mm = 56  # Example margin in mm
            if ids is not None:
                for i, marker_id in enumerate(ids.flatten()):
                    if marker_id == 0:
                        aruco_corners = np.array(corners[i], dtype=np.int32).reshape(4, 2)
                        aruco_bbox = cv2.boundingRect(aruco_corners)
            if aruco_detected:
                marker_size_px = max(aruco_bbox[2], aruco_bbox[3])
                self.scale_factor = marker_size_px / self.marker_size  # pixels per mm
                margin_px = int(margin_mm * self.scale_factor)
                x_min = max(0, aruco_bbox[0] - margin_px)
                y_min = max(0, aruco_bbox[1] - margin_px)
                x_max = min(width, aruco_bbox[0] + aruco_bbox[2] + margin_px)
                y_max = min(height, aruco_bbox[1] + aruco_bbox[3] + margin_px)
                aruco_mask = np.zeros_like(gray_image, dtype=np.uint8)
                aruco_mask[y_min:y_max, x_min:x_max] = 1

                def is_inside_mask(contour, mask):
                    inside_count = sum(1 for point in contour if mask[point[0][1], point[0][0]])
                    return inside_count / len(contour) > 0.5

                contours_filtered = [cnt for cnt in contours_sorted if not is_inside_mask(cnt, aruco_mask)]

            # Draw contours on an overlay image
            overlay = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            if len(contours_filtered) > 0:
                cv2.drawContours(overlay, [contours_filtered[0]], -1, (0, 0, 255), 10)
            if len(contours_filtered) > 1:
                cv2.drawContours(overlay, contours_filtered[1:-1], -1, (0, 255, 0), 2)

            # Stack images for display (original, enhanced, threshold, overlay)
            concatenated = cv2.hconcat([
                undistorted_img,
                cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR),
                overlay
            ])
            concatenated = cv2.resize(concatenated, (0, 0), fx=self.scale, fy=self.scale)
            cv2.imshow("Debug - All Images", concatenated)

            key = cv2.waitKey(50) & 0xFF
            if key == ord('s'):
                # Save current debug parameters to JSON
                with open("Camera Config.json", "w") as f:
                    json.dump({
                        "blur_kernel": blur_kernel,
                        "adaptive_block": adaptive_block,
                        "adaptive_c": adaptive_c,
                        "canny_min": canny_min,
                        "canny_max": canny_max,
                        "min_cont_pixel":min_cont_pixel
                    }, f, indent=4)
                break
            elif key == ord('q'):
                break

        if self.auto:
            cap.release()
        cv2.destroyAllWindows()

    @Logger.log_execution_time("log_obj")
    def extract_contours(self,debug=None):
        
        if debug is None:
            debug=self.debug
        

        # If not in debug mode or after debug adjustments, continue normal processing.
        # Here we will process the image with the chosen parameters (if debug was active) or defaults.
        # Try to load parameters from "Camera Config.json"
        if os.path.exists("Camera Config.json"):
            with open("Camera Config.json", "r") as f:
                loaded_params = json.load(f)
            # Store loaded parameters in self.debug_params (or use them directly)
            self.debug_params = loaded_params
            bk = self.debug_params["blur_kernel"]
            ab = self.debug_params["adaptive_block"]
            ac = self.debug_params["adaptive_c"]
            cmin = self.debug_params["canny_min"]
            cmax = self.debug_params["canny_max"]
            min_cont_pixel=self.debug_params["min_cont_pixel"]
            print("Loaded camera config from JSON.")
        else:
            # If the JSON file doesn't exist, fall back to default values
            print("Camera Config.json not found, using default parameters.")
            bk, ab, ac, cmin, cmax = 23, 11, 2, 100, 200
        if self.auto:
            self.cap=cv2.VideoCapture(0)
            ret, image = self.cap.read()
        else:
            image = cv2.imread(self.picture)
        undistorted_img = cv2.undistort(image, self.camera_matrix, self.dist_coeffs)
        self.image = undistorted_img
        # Convert to grayscale
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray_image, self.aruco_dict, parameters=self.parameters)
        if corners==():
            aruco_detected=False
        else:
            aruco_detected=True
        aruco_mask = np.zeros_like(gray_image)  # Mask to ignore contours inside ArUco box
        
        if ids is not None and 0 in ids:
            index = np.where(ids == 0)[0][0]  # Get index of ID 0
            marker_corners = corners[index].reshape(4, 2).astype(int)
            cv2.fillPoly(aruco_mask, [marker_corners], 255)  # Fill the marker area
            cv2.polylines(image, [marker_corners], True, (0, 0, 255), 3)  # Draw bounding box
        height, width, channels = self.image.shape
        
        image_pixels=height*width
        # Convert to grayscale
        # Apply processing steps
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray_image)
        blurred = cv2.GaussianBlur(enhanced, (bk, bk), 0)
            
        thresh=cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,ab,ac)
        kernel=np.ones((2,2), np.uint8)
        thresh=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)
        edges = cv2.Canny(thresh, cmin, cmax)
            
        # Find contours (excluding ArUco marker area)
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
        while cv2.contourArea(contours_sorted[0])>min_cont_pixel/100*image_pixels:
            contours_sorted.pop(0)
            
        contours_filtered=contours_sorted

        # Define margin in mm
        margin_mm = 56  # Example: 10 mm padding

        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id == 0:  # Only process marker with ID 0
                    aruco_corners = np.array(corners[i], dtype=np.int32).reshape(4, 2)  # Ensure correct shape
                    aruco_bbox = cv2.boundingRect(aruco_corners)  # Get (x, y, width, height)
        if aruco_detected: 
            marker_size_px = max(aruco_bbox[2], aruco_bbox[3])  # Use width or height

            # Compute scale: mm to pixels
            self.scale_factorscale = marker_size_px / self.marker_size  # Pixels per mm

            # Convert margin from mm to pixels
            margin_px = int(margin_mm * self.scale_factorscale)

            # Expand bounding box with converted margin
            x_min = max(0, aruco_bbox[0] - margin_px)
            y_min = max(0, aruco_bbox[1] - margin_px)
            x_max = min(width, aruco_bbox[0] + aruco_bbox[2] + margin_px)
            y_max = min(height, aruco_bbox[1] + aruco_bbox[3] + margin_px)

            # Create exclusion mask
            aruco_mask = np.zeros_like(gray_image, dtype=np.uint8)
            aruco_mask[y_min:y_max, x_min:x_max] = 1

            # Filter contours: Ignore those inside the expanded bounding box
            def is_inside_mask(contour, mask):
                    inside_count = sum(1 for point in contour if mask[point[0][1], point[0][0]])
                    return inside_count / len(contour) > 0.5  # Adjust threshold if needed

                # Remove contours inside the ArUco mask
            contours_filtered = [cnt for cnt in contours_sorted if not is_inside_mask(cnt, aruco_mask)]

        if contours_filtered:
            overlay_image = cv2.cvtColor(edges.copy(), cv2.COLOR_GRAY2BGR)
            cv2.drawContours(overlay_image, [contours_filtered[0]], -1, (0, 255, 0), 2)  # Green color, thickness=2
            
            if debug:
                debug_img=cv2.resize(overlay_image.copy(),(0,0),fx=self.scale,fy=self.scale)
                cv2.imshow("Biggest Contour", debug_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Return contours if needed for further processing
        self.contours_sorted=contours_filtered
        return contours_filtered
    @Logger.log_execution_time("log_obj")
    def extract_eigenvectors(self, debug=None, contours=None):

        if debug is None:
            debug = self.debug

        # Load image for visualization (using self.image if already loaded)
        if self.image is None:
            if self.auto:
                ret, filtered_image = self.cap.read()
            else:
                filtered_image = cv2.imread(self.picture)
            #filtered_image = cv2.resize(filtered_image, (0, 0), fx=self.scale, fy=self.scale)
        else:
            filtered_image = self.image.copy()
        
        # Draw the chosen contour for the object
        if contours is None:
            contours = self.contours_sorted[0]
        cv2.drawContours(filtered_image, [contours], -1, (255, 255, 255), 1)
        
        # Compute moments and object's center-of-gravity (COG)
        M = cv2.moments(contours)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            self.cog=(cX,cY)
        else:
            cX, cY = 0, 0

        # Fit an ellipse to the object (if possible)
        if len(contours) >= 5:
            self.ellipse = cv2.fitEllipse(contours)
            cv2.ellipse(filtered_image, self.ellipse, (0, 0, 255), 3)
            cv2.circle(filtered_image, (cX, cY), 5, (0, 0, 255), -1)
        else:
            print("Not enough points to fit an ellipse.")
        
        # Draw the object's bounding box for reference.
        x, y, w, h = cv2.boundingRect(contours)
        self.bounding_box = {"min": [x, y], "max": [x+w, y+h]}
        cv2.rectangle(filtered_image, (x, y), (x+w, y+h), (0, 255, 255), 2)
        
        # ------------------ ArUco Marker Processing ---------------------
        # Convert image to grayscale for marker detection.
        gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
        
        if ids is not None and 0 in ids:
            # Get marker with ID 0
            index = np.where(ids == 0)[0][0]
            # corners[index] is typically shape (1,4,2); reshape it to (4,2)
            self.marker_corners = np.array(corners[index], dtype=np.int32).reshape(4, 2)
            
            # Compute bounding box of the marker: (x, y, width, height)
            aruco_bbox = cv2.boundingRect(self.marker_corners)
            
            # In the image, (x, y) is the top-left.
            # To define a marker coordinate system with origin at the bottom-left,
            # we set the marker origin (in image coordinates) to the bottom-left of the bounding box.
            self.marker_origin = self.marker_corners[3]  # bottom-left (by the marker's printed order)
            self.x_axis_vec = self.marker_corners[2] - self.marker_origin  # from bottom-left to bottom-right
            self.y_axis_vec = self.marker_corners[0] - self.marker_origin  # from bottom-left to top-left
            
            # For later use, define a helper function to convert from marker coordinates to image coordinates.
            # In the marker coordinate system, the origin is at bottom-left;
            # a point (u, v) in marker coordinates converts to image coordinates as:
            #   (image_x, image_y) = (marker_origin[0] + u, marker_origin[1] - v)
            def marker_to_image(point_marker):
                return np.array([self.marker_origin[0] + point_marker[0],
                                self.marker_origin[1] - point_marker[1]], dtype=np.float32)
            
            # Draw the marker bounding box (still using the original corner points)
            cv2.polylines(filtered_image, [self.marker_corners], True, (255, 0, 255), 3)
            
            # Draw the marker origin (convert self.marker_origin to integer tuple for drawing)
            cv2.circle(filtered_image, tuple(self.marker_origin.astype(int)), 5, (255, 0, 255), -1)
            
            # In the new marker coordinate system:
            # - The x-axis goes from (0, 0) to (width, 0)
            # - The y-axis goes from (0, 0) to (0, height)
            # We know the marker’s bounding box dimensions (width and height) from aruco_bbox.
            marker_width = aruco_bbox[2]
            marker_height = aruco_bbox[3]
            
            # Define the endpoints in marker coordinates.
            x_axis_end_marker = np.array([marker_width, 0], dtype=np.float32)
            y_axis_end_marker = np.array([0, marker_height], dtype=np.float32)
            
            # Convert these endpoints into image coordinates using our helper.
            x_axis_end_img = marker_to_image(x_axis_end_marker)
            y_axis_end_img = marker_to_image(y_axis_end_marker)
            
            # Draw the axes:
            # Red line for the x-axis.
            cv2.line(filtered_image,
                    tuple(self.marker_origin.astype(int)),
                    tuple(x_axis_end_img.astype(int)),
                    (0, 0, 255), 2)
            # Green line for the y-axis.
            cv2.line(filtered_image,
                    tuple(self.marker_origin.astype(int)),
                    tuple(y_axis_end_img.astype(int)),
                    (0, 255, 0), 2)
            
            # Compute pixel-to-mm conversion factor using the known physical marker size.
            # We use the larger dimension of the marker's bounding box.
            marker_pixel_size = max(marker_width, marker_height)
            # self.marker_size should be defined (in mm)
            self.pixel_to_mm = self.marker_size / marker_pixel_size  # mm per pixel
            
            # Suppose (cX, cY) is the object’s center (in image coordinates).
            # We want to compute the vector from the marker’s origin (in the marker coordinate system)
            # to the object’s center.
            # First, convert the object’s image coordinates into marker coordinates.
            # In marker coordinates: 
            #   u = cX - marker_origin[0]
            #   v = marker_origin[1] - cY   (because image y increases downward, but marker y upward)
            vector_marker_to_obj_marker = np.array([cX - self.marker_origin[0],
                                        self.marker_origin[1] - cY], dtype=np.float32)
            
            # Convert that vector length to mm.
            distance_marker_obj_mm = np.linalg.norm(vector_marker_to_obj_marker) * self.pixel_to_mm
            
            # Compute the angle between the marker's x-axis (which is now [1, 0] in marker coordinates)
            # and the vector to the object's center.
            if np.linalg.norm(vector_marker_to_obj_marker) != 0:
                vector_unit = vector_marker_to_obj_marker / np.linalg.norm(vector_marker_to_obj_marker)
            else:
                vector_unit = np.array([1, 0], dtype=np.float32)
            dot_val = np.dot(vector_unit, np.array([1, 0], dtype=np.float32))
            dot_val = np.clip(dot_val, -1.0, 1.0)
            rotation_marker_obj = np.degrees(np.arccos(dot_val))
            
            # Determine the sign of the angle via the cross product.
            # In a right-handed coordinate system with z out of the marker,
            # a positive cross product (z-component) would mean the vector is rotated counterclockwise from x-axis.
            cross = np.cross(np.array([1, 0], dtype=np.float32), vector_unit)
            if cross < 0:
                rotation_marker_obj = -rotation_marker_obj
            
            # Store the transformation parameters in the class.
            self.transformation_translation = vector_marker_to_obj_marker * self.pixel_to_mm  # in mm
            self.transformation_rotation = rotation_marker_obj  # in degrees
            
            # Draw an arrow from the marker origin to the object's COG.
            cv2.arrowedLine(filtered_image, tuple(self.marker_origin.astype(int)), (cX, cY), (0, 0, 0), 2)
            
            # Display transformation information as text.
            info_text = (f"Scale: {self.pixel_to_mm:.2f} mm/px, Distance: {distance_marker_obj_mm:.2f} mm,Rotation: {rotation_marker_obj:.2f} deg")
            print(f"distance from marker to obj: {distance_marker_obj_mm:.2f}")
            cv2.putText(filtered_image, info_text, (10, filtered_image.shape[0] - 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        else:
            print("No ArUco marker with ID 0 detected.")
            # Set default transformation parameters if marker not found
            self.pixel_to_mm = 1.0
            self.transformation_translation = np.array([0, 0], dtype=np.float32)
            self.transformation_rotation = 0.0
        # ------------------ End ArUco Processing ---------------------

        if debug:
            debug_img=cv2.resize(filtered_image.copy(),(0,0),fx=self.scale,fy=self.scale)
            cv2.imshow("Bounding Box, Ellipse, and Axes", debug_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return filtered_image


    @Logger.log_execution_time("log_obj")
    def align_and_transform_paths(self, json_file, debug=False,path_color=(0, 0, 255)):

        # Load the reference header and paths from the JSON file.
        with open(json_file, "r") as f:
            data = json.load(f)
        ref_header = data["header"]
        ref_paths = data["paths"]

        # Extract reference ellipse parameters from JSON.
        # Expected format: [ [center_x, center_y], [axis1, axis2], angle ]
        ref_ellipse = ref_header["ellipse parameters"]
        ref_center = np.array(ref_ellipse[0], dtype=np.float32)
        ref_axes = np.array(ref_ellipse[1], dtype=np.float32)  # full lengths (not half-lengths)
        ref_angle = float(ref_ellipse[2])
        ref_cog=ref_header["center_of_gravity"]

        # Extract real ellipse parameters from the detected shape.
        # Format from cv2.fitEllipse: ((center_x, center_y), (major_axis, minor_axis), angle)
        real_center = np.array(self.ellipse[0], dtype=np.float32)
        real_axes = np.array(self.ellipse[1], dtype=np.float32)
        real_angle = float(self.ellipse[2])
        real_cog=self.cog

        # Compute the required rotation (in radians)
        delta_angle1 = np.deg2rad(real_angle - ref_angle)
        # Candidate 2: add 180° (π radians) to account for ellipse symmetry.
        delta_angle2 = delta_angle1 + np.pi

        R1 = np.array([[np.cos(delta_angle1), -np.sin(delta_angle1)],
                    [np.sin(delta_angle1),  np.cos(delta_angle1)]], dtype=np.float32)
        R2 = np.array([[np.cos(delta_angle2), -np.sin(delta_angle2)],
                    [np.sin(delta_angle2),  np.cos(delta_angle2)]], dtype=np.float32)

        # Compute a uniform scale factor using the average axis length.
        s_ref = np.mean(ref_axes)
        s_real = np.mean(real_axes)
        s = s_real / s_ref

        # Compute the transformation matrix for a point p:
        # T(p) = real_center + s * R * (p - ref_center)
        A1 = s * R1
        A2 = s * R2

        t1 = real_center - np.dot(A1, ref_center)
        t2 = real_center - np.dot(A2, ref_center)

        # --- Choose the candidate transformation that best aligns the centers of gravity ---
        trans_cog1 = np.dot(A1, ref_cog) + t1
        trans_cog2 = np.dot(A2, ref_cog) + t2

        error1 = np.linalg.norm(trans_cog1 - real_cog)
        error2 = np.linalg.norm(trans_cog2 - real_cog)

        if error2 > error1:
            A = A2
            t = t2
        else:
            A = A1
            t = t1
        # Build a 3x3 affine transformation matrix (for homogeneous coordinates)
        T = np.eye(3, dtype=np.float32)
        T[:2, :2] = A
        T[:2, 2] = t

        # Transform each path from the reference coordinate system to the real image (pixel coordinates).
        transformed_paths_json = []
        for path in ref_paths:
            transformed_segments = []

            for segment in path["segments"]:
                transformed_points = []
                for point in segment["points"]:
                    p = np.array([point[0], point[1], 1], dtype=np.float32)
                    p_trans = np.dot(T, p)
                    transformed_points.append([p_trans[0], p_trans[1]])
                
                seg_copy = segment.copy()
                seg_copy["points"] = transformed_points
                transformed_segments.append(seg_copy)


            transformed_paths_json.append({
                "path_id": path["path_id"],
                "segments": transformed_segments
            })


        robot_path = []
        for path in transformed_paths_json:
            for segment in path["segments"]:
                pts = np.array(segment["points"], dtype=np.float32)
                # Convert back to image coordinates:
                pts_image = (pts)  # Inverse transformation
                pts_tuples = tuple(tuple(pt) for pt in pts_image.tolist())
                robot_path.append((segment["type"], pts_tuples))


        # --- Now convert the coordinates from pixel to real-world (mm) with respect to the marker's coordinate system ---
        # We assume that self.pixel_to_mm and self.marker_origin are defined in a previous step.
        if hasattr(self, "pixel_to_mm") and hasattr(self, "marker_origin") and hasattr(self, "x_axis_vec") and hasattr(self, "y_axis_vec"):
            # Marker origin and axes in pixel coordinates
            marker_origin = np.array(self.marker_origin, dtype=np.float32)
            x_axis_vec = np.array(self.x_axis_vec, dtype=np.float32)
            y_axis_vec = np.array(self.y_axis_vec, dtype=np.float32)

            # Calculate the scale factor in both x and y directions from pixel to mm
            scale_factor = self.pixel_to_mm  # mm per pixel

            # Normalize and create a rotation matrix based on the marker's axes
            x_axis_vec = x_axis_vec / np.linalg.norm(x_axis_vec)
            y_axis_vec = y_axis_vec / np.linalg.norm(y_axis_vec)
            rotation_matrix_marker_to_image = np.array([x_axis_vec, y_axis_vec], dtype=np.float32).T

            real_world_paths = []
            for path in transformed_paths_json:
                new_segments = []
                total_length_transformed = 0.0  # Initialize real-world path length
                for segment in path["segments"]:
                    new_points = []
                    for point in segment["points"]:
                        # Convert the point from pixel coordinates to coordinates relative to the marker origin
                        p_pixel = np.array([point[0], point[1]], dtype=np.float32)
                        p_marker = p_pixel - marker_origin
                        # Rotate the point to align with the marker's coordinate system
                        p_marker_rotated = np.dot(rotation_matrix_marker_to_image, p_marker)
                        # Scale the point from pixels to mm
                        p_real = p_marker_rotated * scale_factor
                        new_points.append([p_real[0], p_real[1]])

                    seg_copy = segment.copy()
                    seg_copy["points"] = new_points
                    # Convert the segment's length to real-world mm and accumulate it if present
                    if "length" in seg_copy:
                        seg_copy["length"] = seg_copy["length"] * scale_factor
                        total_length_transformed += seg_copy["length"]
                    new_segments.append(seg_copy)
                real_world_paths.append({
                    "path_id": path["path_id"],
                    "total_length": round(total_length_transformed, 2),
                    "segments": new_segments
                })

            # Update transformed_paths_json to be in real-world coordinates (mm) with total length
            transformed_paths_json = real_world_paths
        else:
            print("Warning: pixel_to_mm, marker_origin, x_axis_vec, or y_axis_vec not set. Returning pixel coordinates.")


        # --- Debug visualization ---
        if debug:
            debug_img = self.image.copy()
            # Draw the real ellipse (in blue).
            cv2.ellipse(debug_img, self.ellipse, (255, 0, 0), 2)

            # Compute transformed reference ellipse parameters rigorously.
            ref_center_trans = np.dot(A, ref_center) + t
            ref_angle_rad = np.deg2rad(ref_angle)
            v_major = np.array([ref_axes[0] / 2, 0], dtype=np.float32)
            v_minor = np.array([0, ref_axes[1] / 2], dtype=np.float32)
            R_ref = np.array([[np.cos(ref_angle_rad), -np.sin(ref_angle_rad)],
                            [np.sin(ref_angle_rad),  np.cos(ref_angle_rad)]], dtype=np.float32)
            v_major = np.dot(R_ref, v_major)
            v_minor = np.dot(R_ref, v_minor)
            v_major_trans = np.dot(A, v_major)
            v_minor_trans = np.dot(A, v_minor)
            new_major = 2 * np.linalg.norm(v_major_trans)
            new_minor = 2 * np.linalg.norm(v_minor_trans)
            new_angle = np.degrees(np.arctan2(v_major_trans[1], v_major_trans[0]))
            ref_center_trans_int = tuple(np.round(ref_center_trans).astype(int))
            axes_half = (int(round(new_major / 2)), int(round(new_minor / 2)))
            cv2.ellipse(debug_img, ref_center_trans_int, axes_half, new_angle, 0, 360, (0, 255, 0), 2)

            # Draw the transformed paths.
            for path in transformed_paths_json:
                for segment in path["segments"]:
                    pts = np.array(segment["points"], dtype=np.float32)
                    # For display purposes, convert back to pixel coordinates by inverting the real-world conversion:
                    # p_pixel = p_mm / scale_factor + marker_origin
                    pts_marker = pts / self.pixel_to_mm  # Undo the scaling (mm -> pixels in the marker's frame)
                    pts_pixel = np.dot(pts_marker, rotation_matrix_marker_to_image.T) + self.marker_origin  # Undo the rotation and translation
                    pts_pixel = np.array(pts_pixel, dtype=np.int32)
                    if segment["type"] == "line":
                        cv2.polylines(debug_img, [pts_pixel], False, path_color, 2)
                    elif segment["type"] == "circ":
                        cv2.polylines(debug_img, [pts_pixel], False, path_color, 2)
                    elif segment["type"] == "point":
                        cv2.circle(debug_img, (pts_pixel[1][0],pts_pixel[1][1]), 2, path_color, 2)
            debug_img=cv2.resize(debug_img,(0,0),fx=self.scale,fy=self.scale)
            cv2.imshow("Alignment Debug", debug_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        print("Length robot path:", len(robot_path))
        return transformed_paths_json, robot_path


    
    def export_contour_as_png(self, contour=None):
        if contour is None:
            contour=self.contours_sorted[0]
        # Optionally, resize or convert the image as needed
        height, width, _ = self.image.shape
        
        # Create a blank canvas (black image)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw the contours on the canvas.
        # You can choose which contours to draw, e.g., the largest one:
        if hasattr(self, "contours_sorted") and len(self.contours_sorted) > 0:
            # Draw all contours or just the largest one
            # For all contours:
            # cv2.drawContours(canvas, self.contours_sorted, -1, (255, 255, 255), 2)
            # For the largest contour:
            cv2.drawContours(canvas, [self.contours_sorted[0]], -1, (255, 255, 255), 2)
        else:
            print("No contours available to export.")
            return

        # Optionally, you could also draw extra information like the bounding box or ellipse if desired.
        cv2.imshow("contour Preview",canvas)
        cv2.waitKey(0)
        # Save the canvas as a PNG file.
        # Create the folder if it doesn't exist
        output_folder = "Imported Shapes"
        os.makedirs(output_folder, exist_ok=True)

        # Ask for user input
        output_filename = input("Please enter the name the shape should be saved as: ")
        output_path = os.path.join(output_folder, output_filename + ".png")

        # Convert the image to grayscale
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

        # Threshold the image to get a binary mask
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Get the bounding box of the largest contour
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])

            # Define a margin
            margin = 10

            # Expand the bounding box with the margin (while keeping it inside image limits)
            x_start = max(0, x - margin)
            y_start = max(0, y - margin)
            x_end = min(binary.shape[1], x + w + margin)
            y_end = min(binary.shape[0], y + h + margin)

            # Crop to the expanded bounding box
            cropped = binary[y_start:y_end, x_start:x_end]

            # Invert the colors (shape black, background white)
            inverted = cv2.bitwise_not(cropped)
            scale_factor=2
            new_width = int(inverted.shape[1] * scale_factor)
            new_height = int(inverted.shape[0] * scale_factor)
            resized = cv2.resize(inverted, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            # Save the result
            success = cv2.imwrite(output_path, resized)

            if success:
                print(f"Contour exported successfully to {output_path}")
            else:
                print("Failed to export the contour.")
        else:
            print("No shape detected to export.")

    def get_intrinsic_camera_parameters(self, file=None, live=True, debug=False):
        """
        Calibrates the camera using a single checkerboard image and saves the intrinsic parameters
        to "Camera Config.json".

        Parameters:
        file (str): Path to the image file to use for calibration (if live is False).
        live (bool): If True, capture one frame from the camera; otherwise, use the image file.
        debug (bool): If True, display the checkerboard detection and overlay calibration parameters.

        Returns:
        dict: A dictionary containing the calibration parameters (camera_matrix, distortion coefficients,
                rotation and translation vectors, image_size), or None on failure.
        """
        # Define the checkerboard parameters:
        # Number of inner corners per chessboard row and column.
        pattern_size = (9, 6)   # Adjust as needed (e.g., 9x6)
        square_size = 24.0      # Size of a square in your chosen units (e.g., millimeters)

        # Prepare object points (0,0,0), (1,0,0), ... scaled by square_size.
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        objp *= square_size

        # Get the calibration image:
        if live:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Cannot open camera.")
                return None
            ret, frame = cap.read()
            cap.release()
            if not ret:
                print("Failed to capture image from camera.")
                return None
            image = frame
        else:
            if file is None:
                print("No file provided for calibration.")
                return None
            image = cv2.imread(file)
            if image is None:
                print("Failed to read image from file.")
                return None

        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if not ret:
            print("Checkerboard pattern not found.")
            return None

        # Refine the corner locations for sub-pixel accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # If debug is enabled, display the detected checkerboard corners
        if debug:
            debug_img = image.copy()
            cv2.drawChessboardCorners(debug_img, pattern_size, corners_refined, ret)
            debug_img=cv2.resize(debug_img,(0,0),fx=0.7,fy=0.7)
            cv2.imshow("Checkerboard Detection", debug_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # For a single image calibration, use lists with one element
        objpoints = [objp]
        imgpoints = [corners_refined]
        image_size = gray.shape[::-1]  # (width, height)

        # Calibrate the camera
        ret_calib, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_size, None, None
        )
        if not ret_calib:
            print("Camera calibration failed.")
            return None

        # Prepare calibration parameters for JSON serialization
        parameters = {
            "camera_matrix": camera_matrix.tolist(),
            "distortion_coefficients": dist_coeffs.tolist(),
            "rotation_vectors": [rvec.tolist() for rvec in rvecs],
            "translation_vectors": [tvec.tolist() for tvec in tvecs],
            "image_size": image_size
        }

        self.dist_coeffs=dist_coeffs
        self.camera_matrix=camera_matrix

        # If debug is enabled, overlay some parameters on the image and show it.
        if debug:
            overlay = image.copy()
            text1 = f"Camera Matrix: {np.round(camera_matrix, 2).tolist()}"
            text2 = f"Dist Coeffs: {np.round(dist_coeffs, 2).tolist()}"
            cv2.putText(overlay, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(overlay, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Calibration Parameters", overlay)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Save calibration parameters to a JSON file
        with open("IntrinsicCameraParameters.json", "w") as f:
            json.dump(parameters, f, indent=4)

        print("Calibration successful. Parameters saved to 'Camera Config.json'.")
        return parameters


    def load_camera_parameters(self, filename="IntrinsicCameraParameters.json"):
        """
        Loads camera calibration parameters from a JSON file and stores them in the object.
        
        Parameters:
        filename (str): Path to the JSON file containing calibration parameters.
        
        Returns:
        dict: The loaded calibration parameters, or None if loading fails.
        """
        try:
            with open(filename, "r") as f:
                parameters = json.load(f)
            self.camera_params = parameters  # For instance, save them to a class attribute.
            print("Camera parameters loaded from", filename)
            return parameters
        except Exception as e:
            print("Error loading camera parameters:", e)
            return None


if __name__ == "__main__":
    '''
    Aligner=AlignShapes(picture=picture)
    file=r"Checkerboard/CheckerboardTestpicture.jpg"
    Aligner.get_intrinsic_camera_parameters(file=file,live=False,debug=True)
    '''
    json_structure=functions.open_json("DecorationShapes/HeartSchreibschrift.json")
    picture="ReferencePictures/W1.jpg"
    Aligner=AlignShapes(json_structure,picture=picture,debug=True,AutoMode=False,scale=1)
    #Aligner.get_intrinsic_camera_parameters(file="Checkerboard/CheckerboardWorkshop.jpg",debug=True,live=False)
    #Aligner.configure_camera()
    Aligner.extract_contours()
    #Aligner.export_contour_as_png()
    Aligner.extract_eigenvectors(debug=True)
    jsonpath,path=Aligner.align_and_transform_paths(json_file="DecorationShapes/Thumbnail.json",debug=True,path_color=(0,0,255))
    #print(jsonpath)
    
    
