# Automated Path Planning for Automated Icing Application

This is my Master Thesis Project to obtain a Master of Science in the field of Robotics at University West, Trollhättan, Sweden.  
It is as shorter GitHub repo where the main aspects are kept intect, but the results might not be perfect. The program runs, but some aspects to improve the paths even further are missing for licencing reasons.
The goal of the practical part of the thesis is to develop an easy method to generate paths for a 6-DoF robot to automatically decorate gingerbreads of any viable shape with icing. The path generation should work for all decorations that are of simple geometric shapes such as lines, circles, or polylines, as well as simple text. The input method should be as easy as possible, requiring no technical knowledge.  
The paths should be sent to a robot and drawn using an icing extruder. To do this, the position of the actual gingerbread needs to be aligned with the decoration, the decoration needs to be transformed to real-world coordinates, and the paths need to be sent to the robot. There, they need to be parsed, the extruder needs to be controlled, and the decoration should be drawn.  

Check out the Youtube Video about the project:

[![Watch the demo](https://img.youtube.com/vi/crPy-hYlLek/maxresdefault.jpg)](https://youtu.be/crPy-hYlLek)
---

## Path Planning

The path planning is implemented using Python and the OpenCV Library.  

### Input Method  

To make the input of the decoration as easy as possible, a picture of the gingerbread the decoration should be applied to can be taken and the contour can be exported. This file can be opened using graphical software such as Paint, and the decoration can be added either using the shapes and text functions or by simply drawing the decorations. The file needs to be saved, and the path can be created.  

### Path Generation  

The picture with the decoration is opened and the decoration is skeletonized. This results in the centerline of each decoration.  
1. The algorithm finds small isolated groups of waypoints. This is usually the case for points such as the dot on the "i".  
2. Next, the algorithm tries to create straight lines on the centerlines of the decoration. The longer the line, the better, since it reduces the amount of data that needs to be sent to the robot.  
3. Finally, the segments are connected with arc segments to create continuous paths that can be drawn by the robot.  

### Local Optimizer  

The segments are still unconnected. Now, a start segment is picked. Ideally, a segment is used that is the beginning of a line (no neighbors on one side, at least one neighbor on the other side). When no such segment can be found, a random segment is selected. The path is expanded based on adjacent segments at the end of the current segment. If there is more than one option, the segment with the most similar direction is selected. This is repeated until all segments are converted into a path.  

### Global Optimizer  

The paths are still in random order. Now, an asymmetrical TSP is solved to find the optimal order of paths to minimize the distance the robot needs to travel between paths. The paths are reordered in the most optimal sequence.  

### Saving the Paths  

The path generation takes a few seconds. Therefore, the paths are stored in a JSON file, including metadata of the shape the decorations are drawn on, as well as metadata on the paths such as length and data on the length of segments and angles.  

## Aligning the Decoration and Gingerbread  

To align the decoration template with the real product, an ellipse is fitted around both shapes. In addition, the center of gravity is calculated to get an unambiguous orientation. The centers of both ellipses are fitted on top of each other, the major and minor axes are scaled, and the two rotations are applied (one shifted by 180°). The version that matches the CoG closest will be selected, and the waypoints created in the path generation with respect to image coordinates are converted into real-world coordinates using an ArUco marker. The usage of ellipses works well, since it reduces the contour points to five parameters (axis lengths, rotation, position, CoG). The template and the real image do not need to align perfectly, but only as well as possible. This allows for small deviations in the real gingerbread that can happen during the baking process.  

## Data Transmission  

The paths are transmitted to the robot via TCP/IP in formatted strings. First, the start coordinate of the path is sent. This is not needed for all subsequent segments since the start position of a segment is the end position of the current segment where the robot is currently positioned. Next, the line type is sent (either point, arc, or line) and the coordinates of the endpoint, and in the case of an arc segment, the via point. In addition, the speed and the needed accuracy are transmitted. The accuracy needs to be higher when the angular difference is very high, indicating a sharp edge. When the lines are mostly aligned, the robot can approximate more.  
Finally, the distance to the next waypoint, where the distance is below a specific threshold to the end of the segment, is defined. This helps in the application of the icing since even if the pump stops, there is still some backflow. This value tells the robot to stop extruding, e.g., 10%, 5%, or 10mm before the path ends. If the last segment, however, is less than this value, the extrusion needs to stop before the last segment.  

## Robot Program  

The robot program establishes a connection between the Python script. A handshake is established to ensure stable communication. Each path is transmitted separately. This allows the robot to start moving toward the approach of the first waypoint after it has been received. After the first path has been fully parsed and saved to the robot's variables, the robot draws the path. At the same time, the next path is transmitted asynchronously. This is needed since the controller of the Kawasaki robot is very slow. With this method, the robot can start drawing while some paths are still transmitted. When the robot draws the paths faster than the transmission and parsing process, the robot waits until the new path has been fully parsed.  
Once a decoration has been drawn, the robot moves back to the home position to allow the camera to capture the new gingerbread and send new coordinates to the robot.  
The pump is controlled proportionally to the robot's speed. A higher speed makes the pump extrude faster to maintain a constant icing flow.  

## Future Works  

The path generation works very well already. The control of the pump would need some extra work. However, it is unclear if the control accuracy that is needed is achievable with the extruder currently in place. In addition, the extruder needs to be close to the gingerbread's surface to allow for accurate icing application. This is especially challenging, since gingerbread has an uneven surface that differs with each gingerbread. One option would be to use a 3D camera to change the Z value in the transmitted paths to accurately match the surface of the gingerbread.  
In addition, the plant needs to be automated even further to make it a viable production plant. This would include a conveyor belt to automate the product flow to the robot. The Python code and the robot program would need to be changed to allow continuous operation.  
To make the control easier for the operators of the plant, a GUI could be implemented that gives a preview of the paths and makes the camera configuration easier.  

## Conclusion  

While it would be nice to have implemented the 3D camera and a more accurate extruder to manage drawing more delicate decorations more accurately, the main focus of the thesis is the proof of work to generate paths for highly customized decorations for small batch sizes with minimal effort. The program can easily be extended with 3D coordinates, and it can be changed to run continuously with minimal effort.  

# Usage  

1. Install the required packages using `pip install -r requirements.txt`.  
2. Get the intrinsic parameters for the used camera using the function located in the Alignment class.  
3. Configure the camera to detect the contour of the shapes. This needs to be done when the lighting conditions change or when a different camera is used.  
4. Export the shape of the gingerbread that should be decorated.  
5. Decorate the gingerbread by opening the template and adding the decorations. Then save the picture as a new file.  
6. Set the picture to the one saved in the step above. Connect the camera and place the gingerbread in the FOV of the camera. The robot and the PC should be connected by Ethernet.  
7. Run the path generation. When a path for that image has already been generated, you will be asked if a new path should be generated.  
8. You get a preview of the decoration on the real gingerbread as captured by the camera. Check if the shape has been detected correctly.  
9. Start the robot program. It creates a server on the robot that looks for clients.  
10. Start the transmission process by pressing any button. The Python program connects to the robot and transmits the path data.  
11. The robot draws the decorations. After it is done, it moves back to the home position.  
