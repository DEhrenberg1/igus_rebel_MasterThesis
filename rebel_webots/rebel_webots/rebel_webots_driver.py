import rclpy
from trajectory_msgs.msg import JointTrajectory
from std_msgs.msg import Float64MultiArray
import threading
import time


class RebelWebotsDriver():
    __velocities = [0.0,0.0,0.0,0.0,0.0,0.0]
    def init(self, webots_node, properties):
        self.__robot = webots_node.robot
        self.__joint1 = self.__robot.getDevice('joint1')
        self.__joint2 = self.__robot.getDevice('joint2')
        self.__joint3 = self.__robot.getDevice('joint3')
        self.__joint4 = self.__robot.getDevice('joint4')
        self.__joint5 = self.__robot.getDevice('joint5')
        self.__joint6 = self.__robot.getDevice('joint6')
        self.__velocities = [0.0,0.0,0.0,0.0,0.0,0.0]
        self.__count = 0
        self.__node = rclpy.create_node('igus_webots_driver')
        self.__node.create_subscription(Float64MultiArray, 'joint_trajectory/command', self.__trajectory_callback, 1)
    
    def __set_vel(self):
        self.__joint1.setPosition(float('inf'))
        self.__joint2.setPosition(float('inf'))
        self.__joint3.setPosition(float('inf'))
        self.__joint4.setPosition(float('inf'))
        self.__joint5.setPosition(float('inf'))
        self.__joint6.setPosition(float('inf'))
        self.__joint1.setVelocity(self.__velocities[0])
        self.__joint2.setVelocity(self.__velocities[1])
        self.__joint3.setVelocity(self.__velocities[2])
        self.__joint4.setVelocity(self.__velocities[3])
        self.__joint5.setVelocity(self.__velocities[4])
        self.__joint6.setVelocity(self.__velocities[5])
        self.__count = 0

    def __trajectory_callback(self, trajectory):
        self.__velocities = trajectory.data
        self.__set_vel()

    def step(self):
        rclpy.spin_once(self.__node, timeout_sec = 0)
        if self.__velocities == [0.0,0.0,0.0,0.0,0.0,0.0]:
            self.__set_vel()
        self.__count = self.__count + 1
        if self.__count % 10 == 0:
            self.__velocities = [0.0,0.0,0.0,0.0,0.0,0.0]

        

def main(args=None):
    rclpy.init(args=args)
    node = RebelWebotsDriver()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()