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
from geometry_msgs.msg import Point
from std_msgs.msg import Bool
import math

class ObjectPositionPublisher():
    def init(self, webots_node, properties):
        self.__webots_node = webots_node
        self.__robot = webots_node.robot
        self.__block1 = self.__robot.getFromDef('block1_solid')
        self.__block2 = self.__robot.getFromDef('block2_solid')
        self.__gripper = self.__robot.getFromDef('gripper')
        self.__distance_gripper_b1 = 0.0
        self.__distance_gripper_b2 = 0.0
        self.__distance_b1_b2 = 0.0
        rclpy.init(args=None)
        self.__node = rclpy.create_node('igus_webots_driver')
        self.__pub = self.__node.create_publisher(Point, '/distance', 10)
        self.__pub1 = self.__node.create_publisher(Point, '/position/block1', 10)
        self.__pub2 = self.__node.create_publisher(Point, '/position/block2', 10)
        self.__pub3 = self.__node.create_publisher(Point, '/position/gripper', 10)
        self.__node.create_subscription(Bool, '/reset', self.__reset_callback, 10)

    def publish_position(self, object, publisher):
        position = object.getPosition()
        msg = Point()
        msg.x = position[0]
        msg.y = position[1]
        msg.z = position[2]
        publisher.publish(msg)
    
    def __compute_distance(self):
        self.__distance_gripper_b1 = math.sqrt((self.__block1.getPosition()[0] - self.__gripper.getPosition()[0])**2 + (self.__block1.getPosition()[1] - self.__gripper.getPosition()[1])**2 + (self.__block1.getPosition()[2] - self.__gripper.getPosition()[2])**2)
        self.__distance_gripper_b2 = math.sqrt((self.__block2.getPosition()[0] - self.__gripper.getPosition()[0])**2 + (self.__block2.getPosition()[1] - self.__gripper.getPosition()[1])**2 + (self.__block2.getPosition()[2] - self.__gripper.getPosition()[2])**2)
        self.__distance_b1_b2 = math.sqrt((self.__block1.getPosition()[0] - self.__block2.getPosition()[0])**2 + (self.__block1.getPosition()[1] - self.__block2.getPosition()[1])**2 + (self.__block1.getPosition()[2] - self.__block2.getPosition()[2])**2)

    def __publish_distance(self):
        msg = Point()
        msg.x = self.__distance_b1_b2
        msg.y = self.__distance_gripper_b1
        msg.z = self.__distance_gripper_b2
        self.__pub.publish(msg)
    
    def __reset_callback(self, msg):
        if msg.data:
            self.__robot.getSelf().restartController()
            self.__robot.simulationResetPhysics()
            self.__robot.simulationReset()

    def step(self):
        rclpy.spin_once(self.__node, timeout_sec=0)
        self.publish_position(self.__block1, self.__pub1)
        self.publish_position(self.__block2, self.__pub2)
        self.publish_position(self.__gripper, self.__pub3)
        self.__compute_distance()
        self.__publish_distance()

def main(args=None):
    rclpy.init(args=args)
    node = ObjectPositionPublisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
