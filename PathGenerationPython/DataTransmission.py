import socket
import functions
import Logger
import time

class DataTransmitter:
    def __init__(self, json_path=None, ip_address=None, port=None, simulation=False, debug=False,log_obj=None):
        self.path = json_path
        self.ip_address = ip_address
        self.port = port
        self.simulate = simulation
        self.debug = debug
        self.log_obj=log_obj
        self.socket = None  # Server socket
        self.conn = None  # Client connection
        self.addr = None  # Client address
    
    def connect_to_server(self):
        """ Connects to an existing server as a client. """
        if not self.simulate:
            if self.port is None or self.ip_address is None:
                print("Error: No IP address or port specified.")
                return
            try:
                self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.conn.connect((self.ip_address, self.port))
                if self.debug:
                    print(f"Connected to server at {self.ip_address}:{self.port}")
            except socket.error as e:
                print(f"Error connecting to server: {e}")
                self.conn = None
        else:
            print("Simulation mode: No connection made.")
    
    def start_server(self):
        """ Starts a socket server that listens for a single client connection. """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.bind((self.ip_address, self.port))
            self.socket.listen(1)  # Only allow 1 client
            if self.debug:
                print(f"Server listening on {self.ip_address}:{self.port}")
            
            self.conn, self.addr = self.socket.accept()
            if self.debug:
                print(f"Accepted a connection from {self.addr}")

        except socket.error as e:
            print(f"Server error: {e}")
        finally:
            self.disconnect_socket()

    def disconnect_socket(self):
        """ Disconnects both the server and the client sockets. """
        if self.conn:
            self.conn.close()
            self.conn = None
        if self.socket:
            self.socket.close()
            self.socket = None
        if self.debug:
            print("Sockets closed.")

    def receive_data(self):
        """ Receives data from the connected client. """
        try:
            data = self.conn.recv(1024)
            if not data:  # Properly checks for disconnection
                print("Connection terminated by client")

                exit(1)
            if self.debug:print("Received Data",data)
            if self.debug:print("Received Decoded Value",data.decode('utf-8'))
            return data.decode('utf-8')
        except socket.error as e:
            print(f"Error receiving data: {e}")
            return None

    def send_data(self, data):
        """ Sends data to the connected client. """
        if self.simulate:
            if self.debug:
                print("Simulation mode: Data not sent.")
            return
        
        if not self.conn:
            print("No active connection to send data.")
            return
        
        try:
            if isinstance(data, str):
                data = data.encode()  # Convert string to bytes
            
            self.conn.sendall(data)
            if self.debug:
                print(f"Sent data: {data}")
        
        except (socket.error, BrokenPipeError) as e:
            print(f"Error sending data: {e}")
            self.disconnect_socket()

    def send_decoration(self, json_path,height=7,speed=40):
        """
        :param json_path: JSON structure containing the paths to be sent to the robot.
        :param height: Height at which the decoration should be printed in mm.
        Sends paths to the robot. Each path starts with `S/(x0/y0/z0)#` indicating a new path.
        Each segment is defined as: "#Type/(x1/y1/z1/x2/y2/z2)/speed in mm/s/accuracy/end_of_line_distance#".
        Example: ": #L/(200/100/20///)/20/5".
        - Type: L (line) or C (circular).
        - x2, y2, z2 are only needed for circular segments.
        - Path ends with `#E`.
        The robot must acknowledge readiness with "ACK".
        """

        # Start communication and wait for ACK
        if self.log_obj:
            transmission_start = time.time()
        if not self.simulate:
            self.send_data("Start")
        else:
            print("Simulate sending Start")
        answere = ""
        while answere != "ACK":
            if self.simulate:
                answere = "ACK"
            else:
                answere = self.receive_data()
                print(answere)

        if self.debug:
            print("Initial ACK for start received")
        answere = ""

        # Handle different JSON formats
        if isinstance(json_path, dict):  # If a single dict, wrap it in a list
            json_paths = [json_path]
        elif isinstance(json_path, list):  # If a list, assume it's already structured
            json_paths = json_path
        else:
            print("Error: Unsupported JSON format")
            return

        if self.debug:
            print(json_paths)  # Debug output

        # For each path...
        print(json_paths[0])
        i=0
        for path in json_paths:
            # Start point command
            start_path = path["segments"][0]
            start_points = start_path["points"]
            x0, y0 = start_points[0]
            command = f"S/({x0:.1f}/{y0:.1f}/{height:.1f})#"

            current_length = 0
            path_length = path["total_length"]
            offset_percentage = 0.02  # For 10% offset.
            offset_length = path_length * offset_percentage

            # Compute the extrusion threshold: extrude until (total length - min_length)
            extrude_threshold = max(path_length - offset_length, 0)
            extruder_off = False  # Flag indicating whether the extruder has been turned off

            # Process each segment
            for segment in path["segments"]:
                seg_length = segment["length"]

                # Convert NumPy floats to regular Python floats if necessary.
                points = segment["points"]
                points = [[float(coord) for coord in point] for point in points]

                # Build the command depending on the segment type.
                if segment["type"] == "line" :
                    # For a line segment, assume two points.
                    x1, y1 = points[0]  # Start point (could be used if needed)
                    x2, y2 = points[1]  # End point
                    # The stop_variable is added as the last parameter
                    command += f"L/({x2:.1f}/{y2:.1f}/{height:.1f}///)/"
                elif segment["type"] == "circ":
                    if len(points) < 3:
                        print(f"Error: Circular segment missing a third point: {segment}")
                        continue
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    x3, y3 = points[2]
                    command += f"C/({x2:.1f}/{y2:.1f}/{height:.1f}/{x3:.1f}/{y3:.1f}/{height:.1f})/"
                elif segment["type"] == "point":
                    x1, y1 = points[0]  # Start point (could be used if needed)
                    x2, y2 = points[1]  # End point
                    command += f"P/({x2:.1f}/{y2:.1f}/{height:.1f}///)/"
                else:
                    print(f"Unknown segment type: {segment['type']}")
                    continue

                # Append fixed speed (here, 5) and accuracy based on the angle difference.
                command += f"{speed}/"

                angle = segment.get("angle_difference", 0)
                if angle is None:
                    accuracy = 1
                elif angle < 20:
                    accuracy = 15
                elif angle < 60:
                    accuracy = 6
                elif angle < 90:
                    accuracy = 2
                else:
                    accuracy = 1
                command += f"{accuracy}/"
                # Determine stop_variable (offset) for this segment:
                # - If the extruder is still on and this segment crosses the threshold,
                #   compute the distance from the segment's end where extruding stops.
                # - Otherwise, the value is 0.
                if not extruder_off:
                    if current_length + seg_length < extrude_threshold:
                        stop_variable = 0
                        command += "0#"
                    else:
                        # This segment crosses the extrusion threshold.
                        # Compute how far into this segment the threshold is reached.
                        distance_extruded_in_seg = extrude_threshold - current_length
                        # The offset (stop_variable) is the remaining part of this segment
                        # that should not extrude.
                        stop_variable = seg_length - distance_extruded_in_seg
                        extruder_off = True  # Pump turns off at this point.
                        command += f"{stop_variable:.1f}#"
                else:
                    stop_variable = 0
                    command += "0#"

                
                # Update the cumulative length along the path.
                current_length += seg_length

            command += "E#"

            # Debug and send the command to the robot.
            if self.simulate:
                print("Sending the Command: ", command)
                if self.log_obj:
                    path_start = time.time()
                print("Simulate receiving ACK")
            else:
                if self.debug:
                    print(f"The length of the string is {len(command)}")
                answere = "-1"
                self.send_data(command)
                if self.log_obj:
                    path_start = time.time()
                while answere != "ACK":
                    answere = self.receive_data()
                answere = ""
            if self.log_obj:
                path_end = time.time()
                self.log_obj.add_entry(f"Path {i} Transmission and parsing of {len(command)} chars", path_end - path_start)
                i+=1


        # End communication
        if self.simulate:
            print("All the paths have been sent")
        else:
            self.send_data("End")
            while answere != "ACK":
                answere = self.receive_data()

        if self.log_obj:
            total_transmission_time = time.time() - transmission_start
            self.log_obj.add_entry("Total Transmission", total_transmission_time)

        
if __name__ == "__main__":
    json_structure=functions.open_json("DecorationShapes/HeartSchreibschrift.json")
    Transmitter=DataTransmitter(json_structure,debug=True,simulation=True,ip_address="192.168.0.2",port=8192)
    Transmitter.connect_to_server()
    Transmitter.send_decoration(json_structure,height=6)