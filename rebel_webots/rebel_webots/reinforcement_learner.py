import gymnasium as gym
import numpy as np
import rclpy
from gymnasium import spaces
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


class ReinforcementLearnerEnvironment(gym.Env):
    def __init__(self):
        super().__init__
        ##OpenAI Stuff
        self.action_space = spaces.Box(low = -1.0, high = 1.0, shape = (1,1), dtype=np.float32)

        ## ROS2 Stuff
        self.__node = rclpy.create_node('reinforcement_learner')
        self.__node.create_subscription(JointState, '/joint_states', self.__joint_state_callback, 1)
        self.__node.create_subscription(Point, '/position/block1', self.__pos_block1_callback, 1)
        self.__node.create_subscription(Point, '/position/block2', self.__pos_block2_callback, 1)
        self.__node.create_subscription(Point, '/position/gripper', self.__pos_gripper_callback, 1)
        self.__arm_publisher = self.__node.create_publisher(Float64MultiArray, '/rebel_arm_controller/commands', 10)
        self.__gripper_publisher = self.__node.create_publisher(JointTrajectory, '/gripper_driver/command', 10)

    ## Rewriting step and reset function of OpenAIGym for our Use Case:
    def step(self, action):
        observation = None
        reward = None
        terminated = None
        truncated = None
        info = None
        return observation, reward, terminated, truncated, info
    
    def reset (self, seed=None, options=None):
        observation = None
        info = None
        return observation, info
    
    #Not sure if these functions are needed.
    #def render(self):
    #    pass

    #def close(self):
    #    pass

    ## The Functions below are used for ROS2-Communication:
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
    
    def move_arm(self, velocities):
        msg = Float64MultiArray()
        msg.data = velocities
        self.__arm_publisher.publish(msg)
    
    def move_gripper(self, pos):
        msg = JointTrajectory()
        msg.joint_names = ['left_finger_joint']
        point = JointTrajectoryPoint()
        point.positions = [pos]
        msg.points.append(point)
        self.__gripper_publisher.publish(msg)



def main(args = None):
    rclpy.init(args=args)
    rle = ReinforcementLearnerEnvironment()
    while(True):
        rle.move_arm([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        rle.move_gripper(0.5)

if __name__ == '__main__':
    main()