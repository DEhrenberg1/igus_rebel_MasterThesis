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
import numpy as np
from geometry_msgs.msg import Point
from std_msgs.msg import Bool
from std_msgs.msg import Float64
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
        self.__pinch_pos = [0.0, 0.0, 0.0]
        rclpy.init(args=None)
        self.__node = rclpy.create_node('igus_webots_driver')
        self.__pub = self.__node.create_publisher(Point, '/distance', 10)
        self.__pub1 = self.__node.create_publisher(Point, '/position/block1', 10)
        self.__pub2 = self.__node.create_publisher(Point, '/position/block2', 10)
        self.__pub3 = self.__node.create_publisher(Point, '/position/gripper', 10)
        self.__pub4 = self.__node.create_publisher(Float64, '/sim_time', 10)
        self.__node.create_subscription(Bool, '/reset', self.__reset_callback, 10)

    def publish_position(self, position, publisher):
        msg = Point()
        msg.x = position[0]
        msg.y = position[1]
        msg.z = position[2] - 0.1
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
    
    def __publish_simulation_time(self):
        sim_time_message = Float64()
        sim_time_message.data = self.__robot.getTime()
        self.__pub4.publish(sim_time_message)
    
    def __reset_callback(self, msg):
        if msg.data:
            x_1 = np.random.uniform(0.4, 0.6)
            y_1 = np.random.uniform(-0.25, 0.25)

            translation_b1 = self.__block1.getField('translation')
            translation_b1.setSFVec3f([x_1, y_1, (0.0275 + 0.1)])
            # Always rotate in direction of (0,0,0) for better grasping chances:
            angle = np.arctan((y_1/x_1))
            rotation_b1 = self.__block1.getField('rotation')
            rotation_b1.setSFRotation([0, 0, 1, angle])

            translation_b2 = self.__block2.getField('translation')
            translation_b2.setSFVec3f([0.8, 0.0, 0.0275])
            rotation_b2 = self.__block2.getField('rotation')
            rotation_b2.setSFRotation([0, 0, 1, 0])

            igus_rebel = self.__robot.getFromDef('igus_rebel')

            j_1 = np.random.uniform(-1.0, 1.0)
            j_2 = np.random.uniform(0.2, 0.5)
            j_3 = np.random.uniform(0.1, 0.3)
            j_4 = np.random.uniform(0.7, 1.4)

            joint_reset_pos = [j_1, j_2, j_3, 0.0, j_4 , 0.0]
            for i in range(6):
                hingejoint = igus_rebel.getFromProtoDef('hingejoint_' + str(i))
                hingejoint.setJointPosition(joint_reset_pos[i])

            for i in range(8):
                hj = self.__gripper.getFromProtoDef('hj' + str(i))
                hj.setJointPosition(0)
            #self.__robot.getSelf().restartController()
            self.__robot.simulationResetPhysics()
            #self.__robot.simulationReset()
    
    def __compute_gripper_pinch_pos(self):
        ##This function computes roughly the pinch position of the fingers (i.e. the position the fingers would meet when closing)
        ##This is based on the position and orientation of the gripper (see webots documentation) and gripper specification
        pos = self.__gripper.getPosition()
        orientation = self.__gripper.getOrientation()
        orientation = np.reshape(orientation, (3,3))
        offset = np.array([0,0,0.085]) #pinch position is roughly 15cm from gripper pos in z-direction in gripper coordinate system
        self.__pinch_pos = np.matmul(orientation,offset) + pos

    def step(self):
        rclpy.spin_once(self.__node, timeout_sec=0)
        pos = self.__block1.getPosition()
        pos[2] = pos[2] + 0.0275
        self.publish_position(pos, self.__pub1)
        self.publish_position(self.__block2.getPosition(), self.__pub2)
        self.__compute_gripper_pinch_pos()
        self.publish_position(self.__pinch_pos, self.__pub3)
        #self.publish_position(self.__gripper.getPosition(), self.__pub3)
        self.__compute_distance()
        self.__publish_distance()
        self.__publish_simulation_time()

def main(args=None):
    rclpy.init(args=args)
    node = ObjectPositionPublisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
