import rclpy
import numpy as np
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from tf_transformations import quaternion_from_euler

class MultiFramePublisher(Node):

    def __init__(self):
        super().__init__('tf_broadcaster')
        self.br = TransformBroadcaster(self)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        self.publish_transform('simulation_frame', 'base_link', 0.0, 0.0, 0.04, 0, 0, 0)
        self.publish_transform('camera_frame', 'simulation_frame', 0.0, 0.064, 1.013, 0.0, (np.pi)/2, -(np.pi)/2)

    def publish_transform(self, parent_frame, child_frame, x, y, z, roll, pitch, yaw):
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame

        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z

        quat = quaternion_from_euler(roll, pitch, yaw)
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.br.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = MultiFramePublisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
