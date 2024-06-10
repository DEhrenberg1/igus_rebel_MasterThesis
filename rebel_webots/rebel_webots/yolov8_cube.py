from ultralytics import YOLO
import cv2
import pykinect_azure as pykinect
from collections import defaultdict
import numpy as np
import rclpy
from geometry_msgs.msg import Point

# Initialize the library, if the library is not found, add the library path as argument
pykinect.initialize_libraries()

# Modify camera configuration
device_config = pykinect.default_configuration
device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_UNBINNED
device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_15


# Start device
device = pykinect.start_device(config=device_config)

calibration = device.get_calibration(device_config.depth_mode, device_config.color_resolution)

#ROS2 Stuff to publish coordinates
rclpy.init(args=None)
node = rclpy.create_node('azure_kinect')
cube_publisher = node.create_publisher(Point, '/real/position/block1', 10)
gripper_publisher = node.create_publisher(Point, '/real/position/gripper', 10)

def publish_position(position, publisher):
    msg = Point()
    msg.x = position[0]
    msg.y = position[1]
    msg.z = position[2]
    publisher.publish(msg)

#Define Yolov8 Model
model = YOLO("best_nano_gripper.pt")

while True:

    # Get capture
    capture = device.update()
    
    # Get the color image from the capture
    ret_color, color_image = capture.get_color_image()

    # Get the depth, mapped to size of the color resolution
    ret_depth, transformed_depth_image = capture.get_transformed_depth_image()

    if not ret_color or not ret_depth:
        continue

    results = model.track(source = color_image, tracker = "bytetrack.yaml", persist = True)
    #Check if an Object was tracked
    # if results[0].boxes.is_track:
    #     x_center = int(results[0].boxes.xywh[0][0])
    #     y_center = int(results[0].boxes.xywh[0][1])
    #     x_lower_left = int(results[0].boxes.xyxy[0][0])
    #     y_lower_left = int(results[0].boxes.xyxy[0][1])
    #     #Get class name:
    #     class_id = int(results[0].boxes.cls[0])
    #     class_label = model.names[class_id]
    #     print(class_label)        
    #     #Compute depth
    #     depth = transformed_depth_image[y_center][x_center]
    #     #Compute X,Y,Z components 
    #     x_irl = (depth*(x_center - calibration.color_params.cx))/calibration.color_params.fx
    #     y_irl = (depth*(y_center - calibration.color_params.cy))/calibration.color_params.fy
    #     z_irl = depth
    #     #Print X,Y,Z components
    #     annotated_frame = results[0].plot()
    #     #annotate frame with coordinates
    #     coordinate_string = "(" + str(int(x_irl)) + ", " + str(int(y_irl)) + ", " + str(int(z_irl)) + ")"
    #     annotated_frame = cv2.putText(annotated_frame, coordinate_string, (x_lower_left,y_lower_left-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    #     # Display the annotated frame
        
        # Check if any objects were tracked
    if results[0].boxes.is_track:
        highest_confidence_per_class = defaultdict(lambda: {'confidence': 0, 'coordinates': (0, 0, 0, 0, 0, 0), 'class_label': ''})

        # Loop through all detected boxes
        for i in range(len(results[0].boxes.xywh)):
            # Extract information for each tracked box
            x_center = int(results[0].boxes.xywh[i][0])
            y_center = int(results[0].boxes.xywh[i][1])
            x_lower_left = int(results[0].boxes.xyxy[i][0])
            y_lower_left = int(results[0].boxes.xyxy[i][1])
            x_upper_right = int(results[0].boxes.xyxy[i][2])
            y_upper_right = int(results[0].boxes.xyxy[i][3])
            
            # Get class ID for the tracked box
            class_id = int(results[0].boxes.cls[i])
            
            # Get the class label for the detected object
            class_label = model.names[class_id]
            
            # Get the confidence score for the detected object
            confidence = results[0].boxes.conf[i]
            
            # Check if this confidence is higher than the current highest for this class
            if confidence > highest_confidence_per_class[class_id]['confidence']:
                highest_confidence_per_class[class_id] = {
                    'confidence': confidence,
                    'coordinates': (x_center, y_center, x_lower_left, y_lower_left, x_upper_right, y_upper_right),
                    'class_label': class_label
                }

        ## Print X,Y,Z components:

        annotated_frame = results[0].plot()

        for class_id, data in highest_confidence_per_class.items():
            x_center = data['coordinates'][0]
            y_center = data['coordinates'][1]
            x_lower_left = data['coordinates'][2]
            y_lower_left = data['coordinates'][3]
            x_upper_right = data['coordinates'][4]
            y_upper_right = data['coordinates'][5]
            confidence = data['confidence']
            class_label = data['class_label']


            # Extract the depth values within the bounding box
            height = y_upper_right - y_lower_left
            if class_label == 'gripper':
                depth_values = transformed_depth_image[(y_lower_left+int(height/3)+5):(y_upper_right-5), (x_lower_left+5):(x_upper_right-5)]
            else:
                depth_values = transformed_depth_image[(y_lower_left+5):(y_upper_right-5), (x_lower_left+5):(x_upper_right-5)]
            depth_values = depth_values[depth_values>0]

            # Compute the average depth
            average_depth = np.mean(depth_values)

            #Compute depth
            depth = transformed_depth_image[y_center][x_center]
            #Compute X,Y,Z components 
            x_cam = (average_depth*(x_center - calibration.color_params.cx))/calibration.color_params.fx
            y_cam = (average_depth*(y_center - calibration.color_params.cy))/calibration.color_params.fy
            z_cam = average_depth

            #Transpose in coordinate system of igus rebel arm (as in simulation)
            x_irl = (-1) * x_cam
            y_irl = ((-1) * y_cam) + 54 
            z_irl = ((-1) * z_cam) + 1030

            #add offset per "item"
            if class_label == 'gripper':
                y_irl = y_irl - 30 #should be -30, for safety reasons -60 ##Should not get under 20
                z_irl = z_irl - 10
            if class_label == 'rubiks-cube':
                y_irl = y_irl + 20
                z_irl = z_irl - 30

            #publish coordinates on ROS2 topic:
            if class_label == 'gripper':
                #publish_position([z_irl/1000, x_irl/1000, y_irl/1000], gripper_publisher)
                pass
            if class_label == 'rubiks-cube':
                publish_position([z_irl/1000, x_irl/1000, y_irl/1000], cube_publisher)
                pass

            #annotate frame with coordinates
            coordinate_string = "(" + str(int(z_irl)) + ", " + str(int(x_irl)) + ", " + str(int(y_irl)) + ")"
            annotated_frame = cv2.putText(annotated_frame, coordinate_string, (x_lower_left,y_lower_left-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    else:
        annotated_frame = results[0].plot()
      
    cv2.imshow("YOLOv8 Tracking", annotated_frame)

    # Press q key to stop
    if cv2.waitKey(1) == ord('q'): 
        break



