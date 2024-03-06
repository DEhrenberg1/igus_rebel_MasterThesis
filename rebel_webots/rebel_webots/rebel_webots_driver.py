#  Copyright (c) 2023 Tarik Viehmann
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import rclpy
from trajectory_msgs.msg import JointTrajectory
from geometry_msgs.msg import Point

class RebelWebotsDriver():
    def init(self, webots_node, properties):
        self.__robot = webots_node.robot
        self.__joint1 = self.__robot.getDevice('joint1')
        self.__joint2 = self.__robot.getDevice('joint2')
        self.__joint3 = self.__robot.getDevice('joint3')
        self.__joint4 = self.__robot.getDevice('joint4')
        self.__joint5 = self.__robot.getDevice('joint5')
        self.__joint6 = self.__robot.getDevice('joint6')
        self.__left_finger_joint = self.__robot.getDevice('ROBOTIQ 2F-85 Gripper::left finger joint')
        self.__positions = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        self.__block1 = self.__robot.getFromDef('block1_solid')
        self.__block2 = self.__robot.getFromDef('block2_solid')

        rclpy.init(args=None)
        self.__node = rclpy.create_node('igus_webots_driver')
        self.__pub1 = self.__node.create_publisher(Point, 'position/block1', 10)
        self.__pub2 = self.__node.create_publisher(Point, 'position/block2', 10)
        #self.__node.create_timer(1.0, self.publish_position(self.__block1_position))
        self.__node.create_subscription(JointTrajectory, 'joint_trajectory/command', self.__trajectory_callback, 1)

    def publish_position(self,object, publisher):
        #if position != [float('nan'),float('nan'),float('nan')]:
        position = object.getPosition()
        msg = Point()
        msg.x = position[0]
        msg.y = position[1]
        msg.z = position[2]
        publisher.publish(msg)

    def __trajectory_callback(self, trajectory):
        self.__positions = trajectory.points[-1].positions
        self.__joint1.setPosition(self.__positions[0])
        self.__joint2.setPosition(self.__positions[1])
        self.__joint3.setPosition(self.__positions[2])
        self.__joint4.setPosition(self.__positions[3])
        self.__joint5.setPosition(self.__positions[4])
        self.__joint6.setPosition(self.__positions[5])
        self.__left_finger_joint.setPosition(self.__positions[6])

    def step(self):
        rclpy.spin_once(self.__node, timeout_sec=0)
        self.publish_position(self.__block1, self.__pub1)
        self.publish_position(self.__block2, self.__pub2)

def main(args=None):
    rclpy.init(args=args)
    node = RebelWebotsDriver()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
