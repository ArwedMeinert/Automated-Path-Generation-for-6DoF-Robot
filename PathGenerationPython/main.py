import PathGeneratorFull
import PathGenerationPython.AlignShapes as AlignShapes
import DataTransmission
import Logger


# Load the paths and attributes from the SVG file

# TODO: Datenpunkte aufnehmen für die Dauer die das parsen und übertragen dauert


log=Logger.ExecutionLogger()
file='DecorationShapes/HV.png'

MyGenerator=PathGeneratorFull.Generator(file,debug=0,log_obj=log)
MyGenerator.generatePath(margin=1,min_line_length=10,min_connected_points_to_filter=3)
MyGenerator.optimisePath(method_global=1,neighborhood_search_multiplyer=1)
print(MyGenerator.robot_path)

json_structure=MyGenerator.json_structure
#cv2.imshow("Path",functions.overlay_robot_path(MyGenerator.filtered_image,MyGenerator.robot_path))
#cv2.waitKey(0)

picture="ReferencePictures/AG1.jpg"
Aligner=AlignShapes.AlignShapes(json_structure,picture=None,AutoMode=True,debug=False,scale=1,log_obj=log)
#Aligner.configure_camera()
Aligner.extract_contours()
#Aligner.export_contour_as_png()
Aligner.extract_eigenvectors()
jsonPaths,paths=Aligner.align_and_transform_paths(json_file="DecorationShapes/HV.json",debug=True)
#print(jsonPaths)

Transmitter=DataTransmission.DataTransmitter(json_structure,debug=True,simulation=False,ip_address="192.168.0.2",port=8192,log_obj=log)
Transmitter.connect_to_server()
Transmitter.send_decoration(jsonPaths,height=30,speed=40)

log.print_log()
#log.export_to_csv(filename="Time Logs/HV.csv")
#MyGenerator.simulatePath(image=Aligner.image,path=paths,debug=False,speed=7,timestep=0.03)
