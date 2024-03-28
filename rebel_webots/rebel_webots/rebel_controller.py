import rclpy
import math
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


class RebelController():
    def init(self, webots_node, properties):
        self.__arm_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.__arm_velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.__gripper_x = 0.0
        self.__gripper_y = 0.0
        self.__gripper_z = 0.0
        self.__block1_x = 0.0
        self.__block1_y = 0.0
        self.__block1_z = 0.0
        self.__block2_x = 0.0
        self.__block2_y = 0.0
        self.__block2_z = 0.0
        self.__distance_gripper_b1 = 0.0
        self.__distance_gripper_b2 = 0.0
        self.__distance_b1_b2 = 0.0

        #rclpy.init(args=None)
        self.__node = rclpy.create_node('rebel_controller')
        self.__node.create_subscription(JointState, '/joint_states', self.__joint_state_callback, 1)
        self.__node.create_subscription(Point, '/position/block1', self.__pos_block1_callback, 1)
        self.__node.create_subscription(Point, '/position/block2', self.__pos_block2_callback, 1)
        self.__node.create_subscription(Point, '/position/gripper', self.__pos_gripper_callback, 1)
        self.__arm_publisher = self.__node.create_publisher(Float64MultiArray, '/rebel_arm_controller/commands', 10)
        self.__gripper_publisher = self.__node.create_publisher(JointTrajectory, '/gripper_driver/command', 10)
        self.__pub = self.__node.create_publisher(Point, '/distance', 10)

    def __joint_state_callback(self, joint_state):
        self.__arm_positions = joint_state.position
        self.__arm_velocities = joint_state.velocity
    
    def __pos_block1_callback(self, pos):
        self.__block1_x = pos.x
        self.__block1_y = pos.y
        self.__block1_z = pos.z
    
    def __pos_block2_callback(self, pos):
        self.__block2_x = pos.x
        self.__block2_y = pos.y
        self.__block2_z = pos.z
    
    def __pos_gripper_callback(self, pos):
        self.__gripper_x = pos.x
        self.__gripper_y = pos.y
        self.__gripper_z = pos.z

    def __compute_distance(self):
        self.__distance_gripper_b1 = math.sqrt((self.__block1_x - self.__gripper_x)**2 + (self.__block1_y - self.__gripper_y)**2 + (self.__block1_z - self.__gripper_z)**2)
        self.__distance_gripper_b2 = math.sqrt((self.__block2_x - self.__gripper_x)**2 + (self.__block2_y - self.__gripper_y)**2 + (self.__block2_z - self.__gripper_z)**2)
        self.__distance_b1_b2 = math.sqrt((self.__block1_x - self.__block2_x)**2 + (self.__block1_y - self.__block2_y)**2 + (self.__block1_z - self.__block2_z)**2)

    def __publish_distance(self):
        msg = Point()
        msg.x = self.__distance_b1_b2
        msg.y = self.__distance_gripper_b1
        msg.z = self.__distance_gripper_b2
        self.__pub.publish(msg)
    
    def __move_arm(self, velocities):
        msg = Float64MultiArray()
        msg.data = velocities
        self.__arm_publisher.publish(msg)
    
    def __move_gripper(self, pos):
        msg = JointTrajectory()
        msg.joint_names = ['left_finger_joint']
        point = JointTrajectoryPoint()
        point.positions = [pos]
        msg.points.append(point)
        self.__gripper_publisher.publish(msg)

    def step(self):
        rclpy.spin_once(self.__node, timeout_sec=0)
        rclpy.spin_once(self.__node, timeout_sec=0)
        rclpy.spin_once(self.__node, timeout_sec=0)
        rclpy.spin_once(self.__node, timeout_sec=0)
        ## Hier muss der Reinforcement Learner einen sinnvollen Input geben!!
        self.__compute_distance()
        self.__publish_distance()
        self.__move_arm([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.__move_gripper(0.5)
        

def main(args=None):
    rclpy.init(args=args)
    node = RebelController()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()