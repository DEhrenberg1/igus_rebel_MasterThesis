import rclpy
from rclpy.node import Node
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Point
import time
from threading import Thread
from ultralytics import YOLO
import cv2
import pykinect_azure as pykinect
from collections import defaultdict
import numpy as np

class CoordinateTransformer(Node):

    def __init__(self):
        super().__init__('coordinate_transformer')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def update_position_from_inference(self, inferred_x, inferred_y, inferred_z):
        # Define the point in the camera frame
        point_in_source_frame = PointStamped()
        point_in_source_frame.header.stamp = self.get_clock().now().to_msg()
        point_in_source_frame.header.frame_id = 'camera_frame'
        point_in_source_frame.point.x = inferred_x
        point_in_source_frame.point.y = inferred_y
        point_in_source_frame.point.z = inferred_z

        try:
            # Look up the transform from the source frame to the target frame
            transform = self.tf_buffer.lookup_transform('simulation_frame', 'camera_frame', rclpy.time.Time())

            # Transform the point to the target frame
            point_in_target_frame = do_transform_point(point_in_source_frame, transform)

            return point_in_target_frame.point
        
        except Exception as e:
            self.get_logger().info(f'Could not transform point: {e}')

#Method to publish the inferred positions
def publish_position(position, publisher):
    msg = Point()
    msg.x = position[0]
    msg.y = position[1]
    msg.z = position[2]
    publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    tf_node = CoordinateTransformer()
    
    ## Start spinning tf_node to update tf_buffer
    executor = MultiThreadedExecutor()
    executor.add_node(tf_node)
    Thread(target = executor.spin).start()

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
    node = rclpy.create_node('azure_kinect')
    cube_publisher = node.create_publisher(Point, '/real/position/block1', 10)

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

                #Compute depth
                depth_values = transformed_depth_image[(y_center-5):(y_center+5), (x_center-5):(x_center+5)]
                depth_values = depth_values[depth_values>0] #Exclude wrong depth values (due to transformation)
                if len(depth_values) > 0: #check that there were non-zero values.
                    depth = np.mean(depth_values)
                    #depth = transformed_depth_image[y_center][x_center]
                    #Compute X,Y,Z components 
                    x_cam = (depth*(x_center - calibration.color_params.cx))/calibration.color_params.fx
                    y_cam = (depth*(y_center - calibration.color_params.cy))/calibration.color_params.fy
                    z_cam = depth
                    #Transpose from mm to m
                    x_cam = x_cam/1000
                    y_cam = y_cam/1000
                    z_cam = z_cam/1000

                    #Wait until we get the transformed coordinates
                    sim_point = None
                    while sim_point == None:
                        sim_point = tf_node.update_position_from_inference(x_cam, y_cam, z_cam)

                    #publish coordinates on ROS2 topic:
                    if class_label == 'rubiks-cube':
                        publish_position([sim_point.x, sim_point.y, sim_point.z], cube_publisher)

                        #annotate frame with coordinates
                        coordinate_string = "(" + str(int(sim_point.x*100)) + ", " + str(int(sim_point.y*100)) + ", " + str(int(sim_point.z*100)) + ")"
                        annotated_frame = cv2.putText(annotated_frame, coordinate_string, (x_lower_left,y_lower_left-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        else:
            annotated_frame = results[0].plot()
        
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Press q key to stop
        if cv2.waitKey(1) == ord('q'): 
            break




    # Example usage with inferred positions from an image
    # This part should be replaced with your actual inference code
    inferred_positions = [
        (0.0940933, 0.0141534, 0.52119),
        (0.1, 0.02, 0.5),  # Example additional positions
        (0.05, 0.03, 0.6),
        (0.05, 0.03, 0.6),
        (0.05, 0.03, 0.6),
        (0.0940933, 0.0141534, 0.52119)    ]

    try:
        for pos in inferred_positions:
            time.sleep(0.1)
            tf_node.update_position_from_inference(pos[0], pos[1], pos[2])
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()
