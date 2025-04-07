import functions
import pathGenerator
import PathOptimisation
import GlobalOptimiser
import Exporter
import cv2
from skimage.morphology import skeletonize
import numpy as np
import json
from PIL import Image, PngImagePlugin
from screeninfo import get_monitors
import Logger
from scipy.ndimage import distance_transform_edt


class Generator:
    def __init__(self,picture,debug=0,log_obj=None):
        '''
        performs the complete path generation and optimisation
        
        :param picture: .png picture that should pe converted into a root path
        :param debug: debug level.  0=No debug; 1= Debug at the path generation of the line and arc segments; 
                                    2= Debug at the local optimisation level; 3= Debug at the global optimisation level
                                    4= Debug at the JSON level
        '''

        self.picture=picture
        self.debug=debug
        self.filename=picture.split('.')[0]
        self.robot_path=None
        self.ordered_path=None
        self.imported=False
        self.log_obj=log_obj

    @Logger.log_execution_time("log_obj")
    def generatePath(self,margin=1,min_line_length=13,min_connected_points_to_filter=2):
        '''
        generates Paths with given parameters. First straight lines are generated, after that arc segments connect the remaining points
        
        :param margin: amount of pixels a line is allowed to deviate from the skeleton
        :param min_line_length: minimal line length. Forces the generator to create long lines or arcs
        '''
        img=Image.open(self.picture)
        filepath=img.info.get("JSON Location","No File")
        print(filepath)
        cv2.waitKey(0)
        if filepath !="No File":
            print("A path has already been created for this file at: ", filepath)
            if input(" Should this file be used instead? (Y/N)").upper()=="Y":
                self.importJSON(filepath)
                self.imported=True
                

        image=cv2.imread(self.picture)

        height, width, channels = image.shape
        # Calculate the number of pixels
        num_pixels = height * width

        gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        contours,_=cv2.findContours(gray_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        contours_sorted=sorted(contours,key=cv2.contourArea,reverse=True)

        while cv2.contourArea(contours_sorted[0])>0.8*num_pixels:
            contours_sorted.pop(0)

        largest_area=cv2.contourArea(contours_sorted[0])


        outlines = [contours_sorted[0]]
        i = 1
        while i < len(contours_sorted) and cv2.contourArea(contours_sorted[i]) > largest_area - largest_area * 0.8:
            outlines.append(contours_sorted[i])
            i += 1
        decoration = contours_sorted[i:]

        # Create binary image for decorations
        decorations = np.zeros_like(gray_image)
        cv2.fillPoly(decorations, decoration, 255)
        if not self.imported:
            skeleton=skeletonize(decorations,method='lee')
            skeleton_contours,_=cv2.findContours(skeleton.astype(np.uint8)*255,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        else:
            skeleton=None
        ##cv2.imshow('skeleton',skeleton.astype(np.uint8)*255)
        ##cv2.waitKey(0)
        # Distance transform on decorations
        dist_transform = cv2.distanceTransform(decorations, cv2.DIST_L2, 5)

        # Threshold to extract the centerline
        _, centerline = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, cv2.THRESH_BINARY)
        centerline = cv2.erode(centerline, None, iterations=1)


        # Display the centerline
        ##cv2.imshow('Centerline', centerline)
        ##cv2.waitKey(0)

        filtered_image = np.zeros_like(image)
        ##contours_decorations=functions.extract_ordered_waypoints_from_skeleton(skeleton)
        self.Converter = pathGenerator.Skeleton_to_path(skeleton,margin=margin,min_line_length=min_line_length,debug=False,min_connected_points_to_filter=min_connected_points_to_filter,log_obj=self.log_obj)
        if not self.imported:
            self.remaining_waypoints=self.Converter.remaining_points
            self.robot_path=self.Converter.robot_path
        ##print(remaining_waypoints)
            for contour in self.remaining_waypoints:
                for waypoint in contour:
                    cv2.circle(filtered_image, waypoint, 1, (0, 255, 0), -1)
            
            cv2.drawContours(filtered_image, skeleton_contours, -1, (100, 100, 100), 1)
        ##filtered_image = functions.overlay_robot_path(filtered_image, robot_path, line_thickness=1,debug=False)
        cv2.drawContours(filtered_image, outlines, -1, (255, 255, 255), 1)
        cv2.drawContours(filtered_image, decoration, -1, (0, 255, 0), 1)
        if self.debug==1:
            cv2.imshow("Debug Path planning",filtered_image)
            cv2.waitKey(0)
        M = cv2.moments(contours[1])
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        
        self.ellipse=cv2.fitEllipse(contours[1])
        cv2.ellipse(filtered_image,self.ellipse,(0,0,255),3)
        cv2.circle(filtered_image, (cX, cY), 5, (0, 0, 255), -1)

        x, y, w, h = cv2.boundingRect(contours[1])
        self.bounding_box={"min": [x,y], "max": [x+w,y+h]}
        self.filtered_image=filtered_image.copy()

    @Logger.log_execution_time("log_obj")
    def optimisePath(self,method_global=1,neighborhood_search_multiplyer=3):
        '''
        optimises the path that is stored in the class. First it tries to connect as many line segments in a row until no more segments 
        can be connected, then it solves a tsp with the line paths and the unconnected segments.
        :param method_global: 1: Uses a Heuristic solver for the TSP; 2: Uses a brute force solver for the TSP; 3: Uses a dynamic solver for the TSP, for uses first a heuristic search, then a local search and 5 only local with random start
        :param neighborhood_search_multiplyer: multiplyer for the margin in the path optimisation
        '''
        if self.debug==2: debug=1 
        else: debug=0
        
        self.Optimisation=PathOptimisation.Optimisation(self.robot_path,log_obj=self.log_obj)
        if not self.imported:
            sorted_path,grouped_paths=self.Optimisation.sort_paths(margin=self.Converter.min_line_length*neighborhood_search_multiplyer,image=self.filtered_image,debug=debug)
        if self.debug==2: print("sorted path old ",sorted_path)

        def unpack_paths(nested_paths):
            # Flatten the outermost path lists (remove the first level of nesting)
            return [segment for path in nested_paths for segment in path]
        
        if not self.imported:
            GOptimiser=GlobalOptimiser.GlobalOptimiser(grouped_paths,log_obj=self.log_obj)
            if method_global==1:
                self.ordered_path,best_path,min_distance=GOptimiser.tsp_heuristic_solver()
            elif method_global==2:
                self.ordered_path,best_path,min_distance=GOptimiser.tsp_solver_brute_force()
            elif method_global==3:
                self.ordered_path,best_path,min_distance=GOptimiser.tsp_solver_dynamic()
            elif method_global==4:
                _,best_path,min_distance=GOptimiser.tsp_heuristic_solver()
                self.ordered_path,best_path,min_distance=GOptimiser.tsp_solver_local(best_path)
            elif method_global==5:
                self.ordered_path,best_path,min_distance=GOptimiser.tsp_solver_local()
            else:
                self.ordered_path,best_path,min_distance=GOptimiser.tsp_heuristic_solver()
            self.ordered_path,best_path,min_distance=GOptimiser.tsp_heuristic_solver()
        print("Ordered Paths: ",self.ordered_path)
        self.robot_path=unpack_paths(self.ordered_path)

        if not self.imported:
            self.exportJSON()
        if self.debug==2: print(self.ordered_path)


    def exportJSON(self):
        '''
        exports the path to a json file including all parameters in the path
        
        :return self.json_structure: minimal line length. Forces the generator to create long lines or arcs
        '''
        # Export JSON using your exporter
        self.exporter = Exporter.RobotPathExporter(self.ordered_path, (250, 250), self.filename, 1, self.bounding_box, self.ellipse)
        self.exporter.export_to_json(self.filename + ".json")
        self.json_structure = self.exporter.convert_ordered_paths_to_json_structure(self.ordered_path)
        
        # Open the image
        img = Image.open(self.picture)
        
        # Create a PNG info object and add your metadata
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("JSON Location", self.filename + ".json")
        
        # Define an output file path (you could also overwrite self.picture if desired)
        output_image_path = self.picture
        
        # Save the image with the metadata; note the call to img.save(...) with parameters.
        img.save(output_image_path, "PNG", pnginfo=pnginfo)
        
        # Optionally, re-open the saved image to check that metadata is present.
        img2 = Image.open(output_image_path)
        if self.debug == 4:
            print(img2.info.get("JSON Location", "Not correctly Exported"))
        
        return self.json_structure


    def importJSON(self,JSONfile):
        '''
        imports a json structure from a json file
        
        :param JSONfile: json file in the os where the paths are stored in
        :return self.json_structure: json structure with all path informations
        '''
        self.filename=JSONfile.split('.')[0]
        self.ordered_path,self.json_structure= functions.convert_json_to_grouped_paths(JSONfile)
        self.robot_path=functions.unpack_paths(self.ordered_path)

        return self.json_structure



    def simulatePath(self,image=None,path=None,debug=None,speed=5,timestep=0.1):
        '''
        simulates a path with given parameters
        
        :param image: specifies the image the path should be shown on. If no image is passed the image from the class is used
        :param path: specifies the path that should be used. if no path is specified the path in the class will be used
        :param debug: if true shows debug information
        :param speed: speed the robot moves in (in pixels per second)
        :param timestep: framerate of the simulation
        '''
        if image is None:
            image=self.filtered_image
        if path is None:
            path=self.robot_path
        if debug is None:
            debug=self.debug
        
        resized_image=self.Converter.simulate_robot_path(image,path,debug=debug,speed=speed,time_interval=timestep)
        return True,image
    


if __name__ == "__main__":
    log=Logger.ExecutionLogger()
    file="DecorationShapes/Pertzborn.png"
    MyGenerator=Generator(file,debug=0,log_obj=log)
    MyGenerator.generatePath(margin=1,min_line_length=13,min_connected_points_to_filter=3)
    MyGenerator.optimisePath(method_global=1,neighborhood_search_multiplyer=1)
    MyGenerator.exportJSON()
    cv2.imshow("Path",functions.overlay_robot_path(MyGenerator.filtered_image,MyGenerator.robot_path))
    log.print_log()
    cv2.waitKey(0)
    functions.display_paths(MyGenerator.filtered_image,MyGenerator.json_structure,debug=True)

    #MyGenerator.simulatePath()