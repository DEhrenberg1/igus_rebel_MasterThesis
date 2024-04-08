import gymnasium as gym
import numpy as np
import rclpy
from gymnasium import spaces
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
#from controller import Supervisor
from stable_baselines3.common.logger import configure

##Ros2 Message Types
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


class ReinforcementLearnerEnvironment(gym.Env):
    def __init__(self):
        super().__init__
        ## Constants
        self.__neg_inf = np.finfo(np.float64).min #negative infinity
        self.__pos_inf = np.finfo(np.float64).max #positive infinity
        self.__max_steps = 300
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
        ## OpenAI Stuff
        self.action_space = spaces.Box(low = -1.0, high = 1.0, shape = (6,), dtype=np.float64) #Velocity controller of all six joints
        self.observation_space = spaces.Dict(
            {
                "rebel_arm_position": spaces.Box(self.__neg_inf, self.__pos_inf, shape= (6,), dtype=np.float64),
                "rebel_arm_velocity": spaces.Box(-1.1, 1.1, shape= (6,), dtype=np.float64),
                "distance_to_block" : spaces.Box(0, self.__pos_inf, shape = (1,), dtype=np.float64)
            }
        )
        self.__current_steps = 0
        ## ROS2 Stuff
        self.__node = rclpy.create_node('reinforcement_learner')
        self.__node.create_subscription(JointState, '/joint_states', self.__joint_state_callback, 1)
        self.__node.create_subscription(Point, '/position/block1', self.__pos_block1_callback, 1)
        self.__node.create_subscription(Point, '/position/block2', self.__pos_block2_callback, 1)
        self.__node.create_subscription(Point, '/position/gripper', self.__pos_gripper_callback, 1)
        self.__node.create_subscription(Point, '/distance', self.__distance_callback, 1)
        self.__arm_publisher = self.__node.create_publisher(Float64MultiArray, '/rebel_arm_controller/commands', 10)
        self.__gripper_publisher = self.__node.create_publisher(JointTrajectory, '/gripper_driver/command', 10)
        self.reset_publisher = self.__node.create_publisher(Bool, '/reset', 10)

    ## Rewriting step and reset function of OpenAIGym for our Use Case:
    def step(self, action):
        print(action)
        self.move_arm([action[0], action[1], action[2], action[3], action[4], action[5]])
        self.__current_steps += 1
        ##Condition for when the action is completed
        action_executed = False
        self.__pos_b1_set = False
        self.__pos_b2_set = False
        self.__joint_state_set = False
        self.__pos_gripper_set = False
        self.__distance_set = False
        while not action_executed:
            rclpy.spin_once(self.__node, timeout_sec=0)
            action_executed = self.__pos_b1_set and self.__pos_b2_set and self.__joint_state_set and self.__pos_gripper_set and self.__distance_set

        observation = {"rebel_arm_position": self.__arm_positions, "rebel_arm_velocity": self.__arm_velocities, "distance_to_block": self.__distance_gripper_b1}
        #observation = {"distance_to_block": self.__distance_gripper_b1}

        reward = 1 if self.reached_grasp_pose(0.02,0.02,0.01) else 0
        terminated = False
        truncated = True if (self.__current_steps >= self.__max_steps) else False
        info = {}
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

        observation = {"rebel_arm_position": self.__arm_positions, "rebel_arm_velocity": self.__arm_velocities, "distance_to_block": self.__distance_gripper_b1}
        #observation = {"distance_to_block": self.__distance_gripper_b1}
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
    
    ##Functions used to compute the reward from the observation:

    def compute_gripping_pos_center(self):
        #computes grasping position of gripper above object
        x = self.__block1_x
        y = self.__block1_y
        z = self.__block1_z + 0.14

        return x,y,z
    
    def reached_grasp_pose(self, x_offset, y_offset, z_offset):
        #returns true if a plausible grasp pose is reached (where plausible is defined by the offset)
        x,y,z = self.compute_gripping_pos_center()
        check_x = abs(x - self.__gripper_x) < x_offset
        check_y = abs(y - self.__gripper_y) < y_offset
        check_z = abs(z - self.__gripper_z) < z_offset

        return check_x and check_y and check_z

def main(args = None):
    rclpy.init(args=args)
    env = ReinforcementLearnerEnvironment()
    # The noise object for DDPG
    action_noise = NormalActionNoise(mean=np.zeros(1,), sigma=0.1 * np.ones(1,))
    
    model = DDPG("MultiInputPolicy", env, action_noise=action_noise, verbose=1, learning_rate = 0.001, tau = 0.001, gradient_steps=-1, learning_starts=900, gamma = 1)
    #model = DDPG.load("ddpg_igus_rebel_test")
    #model.set_env(env)
    tmp_path = "/tmp/sb3_log/"
    # set up logger
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])  
    model.set_logger(new_logger)
    model.learn(total_timesteps = 10000, log_interval=1)
    #model.save("ddpg_igus_rebel_test2")

if __name__ == '__main__':
    main()