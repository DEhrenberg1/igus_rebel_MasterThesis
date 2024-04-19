import gymnasium as gym
import numpy as np
import rclpy
from gymnasium import spaces
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
#from controller import Supervisor
from stable_baselines3.common.logger import configure
import time
from threading import Thread

##Ros2 Message Types
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from std_msgs.msg import Float64
##ROS2 Action
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState

class ReinforcementLearnerEnvironment(gym.Env):
    def __init__(self):
        super().__init__
        ## Constants
        self.__neg_inf = np.finfo(np.float64).min #negative infinity
        self.__pos_inf = np.finfo(np.float64).max #positive infinity
        self.__max_steps = 5
        ## Observation variables
        self.__arm_positions = [0.0,0.0,0.0,0.0,0.0,0.0]
        self.__arm_velocities = [0.0,0.0,0.0,0.0,0.0,0.0]
        self.__block1_x = 0.0
        self.__block1_y = 0.0
        self.__block1_z = 0.0
        self.__block2_x = 0.0
        self.__block2_y = 0.0
        self.__block2_z = 0.0
        self.__gripper_x = 0.0
        self.__gripper_y = 0.0
        self.__gripper_z = 0.0
        self.__distance_gripper_b1 = 2.0
        self.__distance_gripper_b2 = self.__pos_inf
        self.__distance_b1_b2 = self.__pos_inf
        self.__pos_b1_set = False
        self.__pos_b2_set = False
        self.__joint_state_set = False
        self.__pos_gripper_set = False
        self.__distance_set = False
        self.__joint_error = []
        ## OpenAI Stuff
        self.action_space = spaces.Box(low = np.array([-1.0,0.0,0.0,0.0]), high = np.array([1.0,1.0,1.0,1.0]), shape = (4,), dtype=np.float64) #Velocity controller of all six joints
        self.observation_space = spaces.Dict(
            {
                "rebel_arm_position": spaces.Box(self.__neg_inf, self.__pos_inf, shape= (4,), dtype=np.float64),
                "position_block_1": spaces.Box(-3, 3, shape = (3,), dtype=np.float64),
                "position_gripper": spaces.Box(-3, 3, shape = (3,), dtype=np.float64),
                "distance_to_block" : spaces.Box(0, self.__pos_inf, shape = (1,), dtype=np.float64)
            }
        )
        self.__current_steps = 0
        ## ROS2 Stuff
        self.__node = rclpy.create_node('reinforcement_learner')
        self.__node.create_subscription(JointState, '/joint_states', self.__joint_state_callback, 1)
        self.__node.create_subscription(JointTrajectoryControllerState, '/rebel_arm_controller/controller_state', self.__controller_state_callback, 1)
        self.__node.create_subscription(Point, '/position/block1', self.__pos_block1_callback, 1)
        self.__node.create_subscription(Point, '/position/block2', self.__pos_block2_callback, 1)
        self.__node.create_subscription(Point, '/position/gripper', self.__pos_gripper_callback, 1)
        self.__node.create_subscription(Point, '/distance', self.__distance_callback, 1)
        self.__pos_publisher = self.__node.create_publisher(JointTrajectory, '/rebel_arm_controller/joint_trajectory', 10)
        self.__arm_publisher = self.__node.create_publisher(Float64MultiArray, '/rebel_arm_controller/commands', 10)
        self.__gripper_publisher = self.__node.create_publisher(JointTrajectory, '/gripper_driver/command', 10)
        self.reset_publisher = self.__node.create_publisher(Bool, '/reset', 10)

    ## Rewriting step and reset function of OpenAIGym for our Use Case:
    def step(self, action):
        if self.__current_steps == 0:
            time.sleep(3)
        print(action)
        #This commented out action moves arm in gripping pos for block 1
        #self.move_arm([-0.8, 0.4, 0.7, 0.0, 0.5, 0.0])
        self.move_arm([action[0], action[1], action[2], 0.0, action[3], 0.0])
        time.sleep(1.1)
        self.__current_steps += 1
        
        # action_executed = self.__pos_b1_set and self.__pos_b2_set and self.__joint_state_set and self.__pos_gripper_set and self.__distance_set
        # while not action_executed:
        #     rclpy.spin_once(self.__node, timeout_sec=0)
        #     action_executed = self.__pos_b1_set and self.__pos_b2_set and self.__joint_state_set and self.__pos_gripper_set and self.__distance_set
        
        # ##Condition for when the action is completed
        # action_executed = False
        # self.__pos_b1_set = False
        # self.__pos_b2_set = False
        # #self.__joint_state_set = False
        # self.__pos_gripper_set = False
        # self.__distance_set = False

        pos_block1 = [self.__block1_x, self.__block1_y, self.__block1_z]
        pos_gripper = [self.__gripper_x, self.__gripper_y, self.__gripper_z]
        rel_arm_pos = [self.__arm_positions[0],self.__arm_positions[1],self.__arm_positions[2],self.__arm_positions[4]]
        rel_arm_vel = [self.__arm_velocities[0], self.__arm_velocities[1], self.__arm_velocities[2], self.__arm_velocities[4]]

        observation = {"position_block_1": pos_block1, "position_gripper": pos_gripper, "rebel_arm_position": rel_arm_pos, "distance_to_block": self.__distance_gripper_b1}
        info = {}

        #Check if wanted position was reached, if not: assume collision and restart.
        print(self.__joint_error)
        for pos in self.__joint_error:
            if abs(pos) > 0.05:
                reward = 0
                terminated = False
                truncated = True
                print("Collision occured!")
                return observation, reward, terminated, truncated, info
        

        reward = 0.1 if self.reached_grasp_pose(0.2,0.2,0.2) else 0 #Hat mit 0.03 auf allen schon fkt. das es reward gab
        reward = 1 if self.reached_grasp_pose(0.1,0.1,0.1) else reward
        reward = 10 if self.reached_grasp_pose(0.03,0.03,0.03) else reward
        #reward = reward + 0.0001*(10-self.__distance_gripper_b1)
        terminated = False
        truncated = True if (self.__current_steps>=self.__max_steps) else False
        if truncated:
            print(self.__current_steps)
        
        return observation, reward, terminated, truncated, info
    
    def reset (self, seed=None, options=None):
        super().reset(seed=seed)
        print("simulation will be reset")
        msg = Bool()
        msg.data = True
        self.reset_publisher.publish(msg)

        self.__arm_positions = [0.0,0.0,0.0,0.0,0.0,0.0]
        self.__arm_velocities = [0.0,0.0,0.0,0.0,0.0,0.0]
        self.__block1_x = 0.0
        self.__block1_y = 0.0
        self.__block1_z = 0.0
        self.__block2_x = 0.0
        self.__block2_y = 0.0
        self.__block2_z = 0.0
        self.__gripper_x = 0.0
        self.__gripper_y = 0.0
        self.__gripper_z = 0.0
        self.__distance_gripper_b1 = 2.0
        self.__distance_gripper_b2 = self.__pos_inf
        self.__distance_b1_b2 = self.__pos_inf
        self.__current_steps = 0

        while(self.__block1_x == 0 and self.__block2_x == 0):
            pass

        pos_block1 = [self.__block1_x, self.__block1_y, self.__block1_z]
        pos_gripper = [self.__gripper_x, self.__gripper_y, self.__gripper_z]
        rel_arm_pos = [self.__arm_positions[0],self.__arm_positions[1],self.__arm_positions[2],self.__arm_positions[4]]
        rel_arm_vel = [self.__arm_velocities[0], self.__arm_velocities[1], self.__arm_velocities[2], self.__arm_velocities[4]]
        observation = {"position_block_1": pos_block1, "position_gripper": pos_gripper, "rebel_arm_position": rel_arm_pos, "distance_to_block": self.__distance_gripper_b1}
        info = {}
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
        self.__joint_state_set = True
    
    def __pos_block1_callback(self, pos):
        self.__block1_x = pos.x
        self.__block1_y = pos.y
        self.__block1_z = pos.z
        self.__pos_b1_set = True
    
    def __pos_block2_callback(self, pos):
        self.__block2_x = pos.x
        self.__block2_y = pos.y
        self.__block2_z = pos.z
        self.__pos_b2_set = True
    
    def __pos_gripper_callback(self, pos):
        self.__gripper_x = pos.x
        self.__gripper_y = pos.y
        self.__gripper_z = pos.z
        self.__pos_gripper_set = True

    def __distance_callback(self, pos):
        self.__distance_b1_b2 = pos.x
        self.__distance_gripper_b1 = pos.y
        self.__distance_gripper_b2 = pos.z
        self.__distance_set = True
    
    def __controller_state_callback(self, msg):
        self.__joint_error = msg.error.positions

    
    def move_arm(self, pos):
        msg = JointTrajectory()
        msg.header.stamp = self.__node.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        
        point = JointTrajectoryPoint()
        point.positions = pos
        point.velocities = []
        point.accelerations = []
        point.effort = []
        point.time_from_start = Duration(sec=1, nanosec=0)
        
        msg.points.append(point)
        
        self.__pos_publisher.publish(msg)

    def move_gripper(self, pos):
        msg = JointTrajectory()
        msg.joint_names = ['left_finger_joint']
        point = JointTrajectoryPoint()
        point.positions = [pos]
        msg.points.append(point)
        self.__gripper_publisher.publish(msg)
    
    ##Functions used to compute the reward from the observation:
    
    def reached_grasp_pose(self, x_offset, y_offset, z_offset):
        #returns true if a plausible grasp pose is reached (where plausible is defined by the offset)
        check_x = abs(self.__block1_x - self.__gripper_x) < x_offset
        check_y = abs(self.__block1_y - self.__gripper_y) < y_offset
        check_z = abs(self.__block1_z - self.__gripper_z) < z_offset

        return (check_x and check_y and check_z)

def updater(node):
    while True:
        rclpy.spin(node)

def main(args = None):
    rclpy.init(args=args)
    env = ReinforcementLearnerEnvironment()
    Thread(target = updater, args = [env._ReinforcementLearnerEnvironment__node]).start() #Spin Node to update values
    #env.send_goal([0.0,1.0,0.0,0.0,0.0,0.0])
    #The noise object for DDPG
    action_noise = NormalActionNoise(mean=np.zeros(4,), sigma=0.05 * np.ones(4,))
    
    model = DDPG("MultiInputPolicy", env, action_noise=action_noise, verbose=1, learning_rate = 0.001, tau = 0.001, learning_starts=50, gamma = 0.99, batch_size=15, buffer_size= 10000, gradient_steps= 10, train_freq = (1, "episode"))
    #model = DDPG.load("ddpg_igus_rebel_test3")
    model.set_env(env)
    tmp_path = "/tmp/sb3_log/"
    # set up logger
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])  
    model.set_logger(new_logger)
    model.learn(total_timesteps = 30000, log_interval=1)
    #test 6 war 0.03 bei allen
    model.save("position_controller")

if __name__ == '__main__':
    main()