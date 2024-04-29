import rclpy
from std_msgs.msg import Float64
from trajectory_msgs.msg import JointTrajectory

class GripperDriver():
    def init(self, webots_node, properties):
        self.__robot = webots_node.robot
        self.__left_finger_joint = self.__robot.getDevice('ROBOTIQ 2F-85 Gripper::right finger joint')
        #rclpy.init(args=None)
        self.__node = rclpy.create_node('gripper_driver')
        self.__node.create_subscription(JointTrajectory, 'gripper_driver/command', self.__trajectory_callback, 1)

    def __trajectory_callback(self, trajectory):
        new_position = trajectory.points[-1].positions[0]
        self.__left_finger_joint.setPosition(new_position)

    def step(self):
        rclpy.spin_once(self.__node, timeout_sec=0)

def main(args=None):
    rclpy.init(args=args)
    node = GripperDriver()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()