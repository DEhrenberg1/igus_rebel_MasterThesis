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
        self.__step_nr = 1
        #self.__left_finger_joint = self.__robot.getDevice('ROBOTIQ 2F-85 Gripper::left finger joint')
        #self.__positions = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        self.__positions = [0.0,0.0,0.0,0.0,0.0,0.0]
        self.__velocities = [0.0,0.0,0.0,0.0,0.0,0.0]
        #rclpy.init(args=None)
        self.__node = rclpy.create_node('igus_webots_driver')
        self.__node.create_subscription(Float64MultiArray, 'joint_trajectory/command', self.__trajectory_callback, 1)
        threading.Thread(target = self.__spin, daemon= False)
        self.__reset_vel_thread = None

    def __spin(self):
        while True:
            #rclpy.spin_once(self.__node, timeout_sec = 0)
            self.__velocities=[1.0,0.0,0.0,0.0,0.0,0.0]
            #print("Test")
            time.sleep(10)
            

    def __trajectory_callback(self, trajectory):
        self.__velocities = trajectory.data
        # self.__joint1.setPosition(float('inf'))
        # self.__joint2.setPosition(float('inf'))
        # self.__joint3.setPosition(float('inf'))
        # self.__joint4.setPosition(float('inf'))
        # self.__joint5.setPosition(float('inf'))
        # self.__joint6.setPosition(float('inf'))
        # self.__joint1.setVelocity(self.__velocities[0])
        # self.__joint2.setVelocity(self.__velocities[1])
        # self.__joint3.setVelocity(self.__velocities[2])
        # self.__joint4.setVelocity(self.__velocities[3])
        # self.__joint5.setVelocity(self.__velocities[4])
        # self.__joint6.setVelocity(self.__velocities[5])
        # self.__joint1.setPosition(self.__positions[0])
        # self.__joint2.setPosition(self.__positions[1])
        # self.__joint3.setPosition(self.__positions[2])
        # self.__joint4.setPosition(self.__positions[3])
        # self.__joint5.setPosition(self.__positions[4])
        # self.__joint6.setPosition(self.__positions[5])
        #self.__left_finger_joint.setPosition(self.__positions[6])

    def step(self):
        #rclpy.spin_once(self.__node, timeout_sec = 0)
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
        #rclpy.spin_once(self.__node, timeout_sec=0)
        pass
        

def main(args=None):
    rclpy.init(args=args)
    node = RebelWebotsDriver()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()