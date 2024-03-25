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

class RebelWebotsDriver():
    def init(self, webots_node, properties):
        self.__robot = webots_node.robot
        self.__block1 = self.__robot.getFromDef('block1_solid')
        self.__block2 = self.__robot.getFromDef('block2_solid')

        rclpy.init(args=None)
        self.__node = rclpy.create_node('igus_webots_driver')
        self.__pub1 = self.__node.create_publisher(Point, 'position/block1', 10)
        self.__pub2 = self.__node.create_publisher(Point, 'position/block2', 10)

    def publish_position(self,object, publisher):
        #if position != [float('nan'),float('nan'),float('nan')]:
        position = object.getPosition()
        msg = Point()
        msg.x = position[0]
        msg.y = position[1]
        msg.z = position[2]
        publisher.publish(msg)

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
