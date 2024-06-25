import rclpy
from rclpy.node import Node
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point

class CoordinateTransformer(Node):

    def __init__(self):
        super().__init__('coordinate_transformer')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        # Define the point in the source frame (e.g., right-hand coordinate system)
        point_in_source_frame = PointStamped()
        point_in_source_frame.header.stamp = self.get_clock().now().to_msg()
        point_in_source_frame.header.frame_id = 'simulation_frame'
        point_in_source_frame.point.x = 0.0940933
        point_in_source_frame.point.y = 0.0141534
        point_in_source_frame.point.z = 0.52119

        try:
            # Look up the transform from the source frame to the target frame
            transform = self.tf_buffer.lookup_transform('simulation_frame', 'camera_frame', rclpy.time.Time())

            # Transform the point to the target frame
            point_in_target_frame = do_transform_point(point_in_source_frame, transform)

            self.get_logger().info(
                f'Transformed Point: x={point_in_target_frame.point.x}, '
                f'y={point_in_target_frame.point.y}, '
                f'z={point_in_target_frame.point.z}')
        except Exception as e:
            self.get_logger().info(f'Could not transform point: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = CoordinateTransformer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()
