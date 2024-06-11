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
from sensor_msgs.msg import JointState


class GripperPositionReal():
    def init(self, webots_node, properties):
        self.__webots_node = webots_node
        self.__robot = webots_node.robot
        self.__gripper = self.__robot.getFromDef('gripper')
        self.__pinch_pos = [0.0, 0.0, 0.0]
        self.arm_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.arm_velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        rclpy.init(args=None)
        self.__node = rclpy.create_node('igus_webots_driver')
        self.__pub = self.__node.create_publisher(Point, '/real/position/gripper', 10)
        self.__node.create_subscription(JointState, '/real/filtered_joint_states', self.joint_state_callback, 1)
        

    def publish_position(self, position, publisher):
        msg = Point()
        msg.x = position[0]
        msg.y = position[1]
        msg.z = position[2]
        publisher.publish(msg)
    
    def joint_state_callback(self, joint_state):
        arm_name = joint_state.name
        self.arm_positions = [] #Positions of the real robot arm
        self.arm_velocities = []
        if len(joint_state.velocity) == 6 and len(joint_state.position) == 6:
            for i in range(6):
                ind = arm_name.index("joint" + str(i+1))
                self.arm_positions.append(joint_state.position[ind])
                self.arm_velocities.append(joint_state.velocity[ind])

            #Set the position of the arm in simuation to the position of the real arm
            igus_rebel = self.__robot.getFromDef('igus_rebel')
            joint_reset_pos = self.arm_positions
            for i in range(6):
                hingejoint = igus_rebel.getFromProtoDef('hingejoint_' + str(i))
                hingejoint.setJointPosition(joint_reset_pos[i])

    
    def __compute_gripper_pinch_pos(self):
        ##This function computes roughly the pinch position of the fingers (i.e. the position the fingers would meet when closing)
        ##This is based on the position and orientation of the gripper (see webots documentation) and gripper specification
        pos = self.__gripper.getPosition()
        pos = (pos[0],pos[1],pos[2] - 0.1) #Offset in real world
        orientation = self.__gripper.getOrientation()
        orientation = np.reshape(orientation, (3,3))
        offset = np.array([0,0,0.15]) #pinch position is roughly 15cm from gripper pos in z-direction in gripper coordinate system
        self.__pinch_pos = np.matmul(orientation,offset) + pos

    def step(self):
        rclpy.spin_once(self.__node, timeout_sec=0)
        self.__compute_gripper_pinch_pos()
        self.publish_position(self.__pinch_pos, self.__pub)
        # self.publish_position(self.__gripper.getPosition(), self.__pub)

def main(args=None):
    rclpy.init(args=args)
    node = GripperPositionReal()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
