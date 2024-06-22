import gymnasium as gym
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from gymnasium import spaces
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from reinforcement_learner import ReinforcementLearnerEnvironment
from rl_grasp_pose import RL_grasp_pose
#from controller import Supervisor
from stable_baselines3.common.logger import configure
import time
import pandas as pd
import threading
from threading import Thread
import math
##Ros2 Message Types
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from std_msgs.msg import Float64

class RL_placing(gym.Env):
    def __init__(self, above_model_name, above_model_env, grasp_model_name, grasp_model_env):
        super().__init__
        ## Constants
        self.__neg_inf = np.finfo(np.float64).min #negative infinity
        self.__pos_inf = np.finfo(np.float64).max #positive infinity
        self.__max_steps = 50
        ## Observation variables
        self.__arm_positions = [0.0,0.0,0.0,0.0,0.0,0.0]
        self.__arm_velocities = [0.0,0.0,0.0,0.0,0.0,0.0]
        self.__block1_x = 0.0
        self.__block1_y = 0.0
        self.__block1_z = 0.0
        self.__block1_initial = [self.__block1_x, self.__block1_y, self.__block1_z]
        self.__block2_x = 0.0
        self.__block2_y = 0.0
        self.__block2_z = 0.0
        self.__gripper_pos = [0.0, 0.0, 0.0] #xyz
        self.__distance_gripper_b1 = 2.0
        self.__distance_gripper_b2 = self.__pos_inf
        self.__distance_b1_b2 = self.__pos_inf
        self.__pos_b1_set = False
        self.__pos_b2_set = False
        self.__joint_state_set = False
        self.__pos_gripper_set = False
        self.__sim_time = 0.0
        self.__reached_pose_above_block = False
        self.__above_model_name = above_model_name
        self.__above_model = DDPG.load(above_model_name, above_model_env)
        self.__grasp_model_name = grasp_model_name
        self.__grasp_model = DDPG.load(grasp_model_name, grasp_model_env)
        self.__safety_distance_ground = 0.03 #The safety distance from the pinch position to the ground in meters

        self.__distance_reward_active = False #Set this to true to activate a small distance based reward
        self.__dont_move_block_active = False #Set this to true if you want the episode to end when the block was moved
        self.__grasp_at_end_active = False #Set this to true if you want to manually try a grasp at the end of the episode if the arm is in a promising position
        self.__start_close_to_goal_active = False #Set this to true to reset so long until the gripper starts close to the object


        ## OpenAI Stuff
        self.action_space = spaces.Box(low = -0.4, high = 0.4, shape = (4,), dtype=np.float64) #Velocity controller of all six joints
        self.observation_space = spaces.Dict(
            {
                "rebel_arm_position": spaces.Box(self.__neg_inf, self.__pos_inf, shape= (4,), dtype=np.float64),
                "rebel_arm_velocity": spaces.Box(-1.1, 1.1, shape = (4,), dtype=np.float64),
                "position_block_2": spaces.Box(-3, 3, shape = (3,), dtype=np.float64),
                "position_gripper": spaces.Box(-3, 3, shape = (3,), dtype=np.float64),
                "distance_to_block" : spaces.Box(0, self.__pos_inf, shape = (1,), dtype=np.float64)
            }
        )
        self.__current_steps = 0
        ## ROS2 Stuff
        self.__node = rclpy.create_node('rl_placing')
        self.__node.create_subscription(JointState, '/joint_states', self.__joint_state_callback, 1)
        self.__node.create_subscription(Point, '/position/block1', self.__pos_block1_callback, 1)
        self.__node.create_subscription(Point, '/position/block2', self.__pos_block2_callback, 1)
        self.__node.create_subscription(Point, '/position/gripper', self.__pos_gripper_callback, 1)
        self.__arm_publisher = self.__node.create_publisher(Float64MultiArray, '/joint_trajectory/command', 10)
        self.__gripper_publisher = self.__node.create_publisher(JointTrajectory, '/gripper_driver/command', 10)
        self.reset_publisher = self.__node.create_publisher(Bool, '/reset', 10)
    
    def limitedAction(self, action):
        lim_act = [action[0], action[1], action[2], 0.0, action[3], 0.0] #limited joint velocities
        joint_lim_max = [1.0,1.0,1.0,0.0,1.5,0.0]
        joint_lim_min = [-1.0,0.1,0.2,0.0,0.3,0.0]
        #Do not exceed the joint position limits specified in joint_lim_max and joint_lim_min:
        for i in range(6):
            if self.__arm_positions[i] > joint_lim_max[i]:
                lim_act[i] = 0.0 if lim_act[i] > 0.0 else lim_act[i]
            if self.__arm_positions[i] < joint_lim_min[i]:
                lim_act[i] = 0.0 if lim_act[i] < 0.0 else lim_act[i]
        
        return lim_act

    ## Rewriting step and reset function of OpenAIGym for our Use Case:
    def step(self, action):
        action = self.limitedAction(action)
        self.move_arm(action)
        self.__current_steps += 1
        if self.__current_steps == 1:
            self.__start_time = self.__sim_time

        self.__distance_gripper_b2 = self.compute_distance_gripper_b2()
        wait_time = 4
        if self.__distance_gripper_b2 <= 0.05:
            wait_time = 3
        if self.__distance_gripper_b2 <= 0.03:
            wait_time = 2
        if self.__distance_gripper_b2 <= 0.02:
            wait_time = 1
        for _ in range(wait_time):
            ##Condition for when the action got executed for one simulation step
            action_executed = False
            self.__pos_b1_set = False
            self.__pos_b2_set = False
            self.__pos_gripper_set = False
            while not action_executed:
                action_executed = self.__pos_b1_set and self.__pos_b2_set and self.__pos_gripper_set

        self.__distance_gripper_b2 = self.compute_distance_gripper_b2()
        pos_block2 = [self.__block1_x, self.__block1_y, self.__block1_z]
        rel_arm_pos = [self.__arm_positions[0],self.__arm_positions[1],self.__arm_positions[2],self.__arm_positions[4]]
        rel_arm_vel = [self.__arm_velocities[0], self.__arm_velocities[1], self.__arm_velocities[2], self.__arm_velocities[4]]
        observation = {"position_block_2": pos_block2, "position_gripper": self.__gripper_pos, "rebel_arm_position": rel_arm_pos, "rebel_arm_velocity": rel_arm_vel, "distance_to_block": self.__distance_gripper_b2}

        reward = 0
        print(self.__distance_gripper_b2)
        if self.reached_place_pose(0.02,0.02,0.02):
            reward = reward + 10
            print("Hurra!")
        if self.reached_place_pose(0.01,0.01,0.01):
            reward = reward + 100
            print("Double Hurra!")
        
        if self.__distance_reward_active:
            if self.__distance_gripper_b2 < 1.0:
                distance_reward = (10 ** (-1 * int(10*self.__distance_gripper_b1))) * 5
                reward = reward + distance_reward
    
        terminated = False
        #End episode after max number of steps
        truncated = True if (self.__current_steps>=self.__max_steps) else False
        #End episode if gripper too close to the ground
        reward = reward - 0.5 if (self.__gripper_pos[2] <= self.__safety_distance_ground) else reward
        truncated = True if (self.__gripper_pos[2] <= self.__safety_distance_ground) else truncated

        if truncated:
            print(self.__current_steps)
        info = {}
        return observation, reward, terminated, truncated, info
    
    def reset (self, seed=None, options=None):

        self.__current_steps = 0
        
        ## Move in position above grasp:
        vec_env = self.__above_model.get_env()
        obs = vec_env.reset()
        done = False
        rewards_0 = 0
        rewards_1 = 0
        rewards_2 = 0
        # wait until 3 consecutive steps in above pose:
        while rewards_0 < 20 and rewards_1 < 20 and rewards_2 < 20:
            action, _states = self.__above_model.predict(obs)
            obs, rewards, done, info = vec_env.step(action)
            rewards_2 = rewards_1
            rewards_1 = rewards_0
            rewards_0 = rewards

        ## Move in grasp position
        vec_env = self.__grasp_model.get_env()
        obs = vec_env.reset()
        done = False
        # wait until in 1cm box around grasp position
        rewards = 0
        while rewards<100:
            action, _states = self.__grasp_model.predict(obs)
            obs, rewards, done, info = vec_env.step(action)
        
        ## Return observations for RL
        self.__distance_gripper_b2 = self.compute_distance_gripper_b2()
        pos_block2 = [self.__block2_x, self.__block2_y, self.__block2_z]

        rel_arm_pos = [self.__arm_positions[0],self.__arm_positions[1],self.__arm_positions[2],self.__arm_positions[4]]
        rel_arm_vel = [self.__arm_velocities[0], self.__arm_velocities[1], self.__arm_velocities[2], self.__arm_velocities[4]]
        observation = {"position_block_2": pos_block2, "position_gripper": self.__gripper_pos, "rebel_arm_position": rel_arm_pos, "rebel_arm_velocity": rel_arm_vel, "distance_to_block": self.__distance_gripper_b2}
        info = {}
        return observation, info

    ## The Functions below are used for ROS2-Communication:
    def __joint_state_callback(self, joint_state):
        arm_name = joint_state.name
        self.__arm_positions = []
        self.__arm_velocities = []
        for i in range(6):
            ind = arm_name.index("joint" + str(i+1))
            self.__arm_positions.append(joint_state.position[ind])
            self.__arm_velocities.append(joint_state.velocity[ind])
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
        self.__gripper_pos = [pos.x, pos.y, pos.z]
        self.__pos_gripper_set = True
    
    def move_arm(self, velocities):
        msg = Float64MultiArray()
        scaled_vel = [x*0.5 for x in velocities]
        msg.data = scaled_vel
        self.__arm_publisher.publish(msg)
    
    def move_gripper(self, pos):
        msg = JointTrajectory()
        msg.joint_names = ['left_finger_joint']
        point = JointTrajectoryPoint()
        point.positions = [pos]
        msg.points.append(point)
        self.__gripper_publisher.publish(msg)

    ##Functions used to compute distance:

    def compute_distance_gripper_b1(self):
        dist = math.sqrt((self.__gripper_pos[0] - self.__block1_x)**2 + (self.__gripper_pos[1] - self.__block1_y)**2 + (self.__gripper_pos[2] - self.__block1_z)**2)
        return dist
    
    def compute_distance_gripper_b2(self):
        dist = math.sqrt((self.__gripper_pos[0] - self.__block2_x)**2 + (self.__gripper_pos[1] - self.__block2_y)**2 + (self.__gripper_pos[2] - self.__block2_z)**2)
        return dist

    def compute_distance_gripper_above_b1(self):
        dist = math.sqrt((self.__gripper_pos[0] - self.__block1_x)**2 + (self.__gripper_pos[1] - self.__block1_y)**2 + ((self.__gripper_pos[2] - 0.05) - self.__block1_z)**2)
        return dist
    
    ##Functions used to compute the reward from the observation:
    
    def reached_pose_above_block(self, x_offset, y_offset, z_offset):
        #returns true if a plausible pose above the grasp pose is reached (where plausible is defined by the offset)
        #This is meant to ensure that the gripper does not move the block sideways
        check_x = abs(self.__block1_x - self.__gripper_pos[0]) < x_offset
        check_y = abs(self.__block1_y - self.__gripper_pos[1]) < y_offset
        check_z = abs(self.__block1_z + 0.05 - self.__gripper_pos[2]) < z_offset

        return (check_x and check_y and check_z)
    
    def reached_place_pose(self, x_offset, y_offset, z_offset):
        #return true if a plausible placing pose is reached (where plausible is defined by the offset)
        check_x = abs(self.__block2_x - self.__gripper_pos[0]) < x_offset
        check_y = abs(self.__block2_y - self.__gripper_pos[1]) < y_offset
        check_z = abs(self.__block2_z + 0.05 - self.__gripper_pos[2]) < z_offset

        return (check_x and check_y and check_z)


    def reached_grasp_pose(self, x_offset, y_offset, z_offset):
        #returns true if a plausible grasp pose is reached (where plausible is defined by the offset)
        check_x = abs(self.__block1_x - self.__gripper_pos[0]) < x_offset
        check_y = abs(self.__block1_y - self.__gripper_pos[1]) < y_offset
        check_z = abs(self.__block1_z - self.__gripper_pos[2]) < z_offset

        return (check_x and check_y and check_z)

def learn_position(model_name, number_of_models, env):
    #Number of total_timesteps:
    high_act_noise_timesteps = 10000
    reduced_act_noise_timesteps_per_iteration = 10000
    low_act_noise_timeteps = 100000
    # The noise object for DDPG
    action_noise = NormalActionNoise(mean=np.zeros(4,), sigma=0.5 * np.ones(4,))
    model = DDPG("MultiInputPolicy", env, action_noise=action_noise, verbose=1, learning_rate = 0.001, tau = 0.001, learning_starts=10000, gamma = 0.99, batch_size=32  , buffer_size= 300000, gradient_steps= 4, train_freq = (1, "episode"))
    model.set_env(env)
    for j in range(number_of_models):
        # set up logger
        path = model_name + str(j) + "log"
        new_logger = configure(path, ["stdout", "csv", "tensorboard"])  
        model.set_logger(new_logger)
        # learn
        model.learn(total_timesteps = high_act_noise_timesteps, log_interval=1)
        # save
        model.save(model_name + str(j))
        model.save_replay_buffer(model_name + str(j))

        #Do not train bad models further:
        df = pd.read_csv(model_name + str(j) + "log" + "/progress.csv")
        reward_column ="rollout/ep_rew_mean"
        last_20_rewards = df[reward_column].tail(20)
        average_reward = last_20_rewards.mean()

        # if average_reward < 1.7:
        #     continue

        #Learn with reduced action noise if model is promising:
        model = None
        for i in range(9):
            action_noise = NormalActionNoise(mean=np.zeros(4,), sigma= 0.25 * (0.9-i/10) * np.ones(4,))
            if i == 0:
                model = DDPG.load(model_name + str(j) + ".zip", learning_starts = 0, action_noise=action_noise)
                model.load_replay_buffer(model_name + str(j) + ".pkl")
            else:
                model = DDPG.load(model_name + str(j) + "_" + str(i-1)+".zip", learning_starts = 0, action_noise=action_noise)
                model.load_replay_buffer(model_name + str(j) + "_" + str(i-1) + ".pkl")
            model.set_env(env)
            # set up logger
            name = model_name + str(j) + "_"  + str(i)
            path = name + "log"
            new_logger = configure(path, ["stdout", "csv", "tensorboard"])  
            model.set_logger(new_logger)
            model.learn(total_timesteps=reduced_act_noise_timesteps_per_iteration, log_interval= 10)
            model.save(name)
            model.save_replay_buffer(name)
        
        model.learn(total_timesteps=low_act_noise_timeteps, log_interval=1)
        model.save(model_name + "final")

def test_model(model, env):
    #model = DDPG.load("exact_position_learner_gui_true_0_8_trained_1_learned_gui_false")
    model.set_env(env)
    vec_env = model.get_env()
    obs = vec_env.reset()
    done = False
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = vec_env.step(action)

def test_grasp_from_position_learner(model, env):
    model.set_env(env)
    vec_env = model.get_env()
    obs = vec_env.reset()
    done = False
    reward = 0
    succeed = 0
    failed = 0
    trials = 0
    while True:
        done = False
        trials = trials + 1
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = vec_env.step(action)
            reward = reward + rewards
            if rewards > 100:
                env.move_gripper(0.8)
                time.sleep(0.4)
                for _ in range(15):
                    lim_act = env.limitedAction([0.0,-0.4,0.0,0.0])
                    env.move_arm(lim_act)
                    time.sleep(0.1)
                if env._RL_grasp_pose__block1_z > 0.1:
                    succeed = succeed + 1
                obs = vec_env.reset()
                reward = 0
                done = True
        print("Success rate: " + str(succeed/trials) + ", Trials: " + str(trials))


lock = threading.Lock()

def updater(node):
    while True:
        rclpy.spin(node)


def main(args = None):
    ## Initialisitation
    rclpy.init(args=args)
    env_above_pose = ReinforcementLearnerEnvironment()
    env_above_pose._ReinforcementLearnerEnvironment__distance_reward_active = True
    env_above_pose._ReinforcementLearnerEnvironment__start_close_to_goal_active = False
    env_above_pose._ReinforcementLearnerEnvironment__dont_move_block_active = False 
    env_above_pose._ReinforcementLearnerEnvironment__grasp_at_end_active = False 

    env_grasp_pose = RL_grasp_pose("get_in_pose_above1_8.zip", env_above_pose)
    env_grasp_pose._RL_grasp_pose__distance_reward_active = True
    env_grasp_pose._RL_grasp_pose__start_close_to_goal_active = False
    env_grasp_pose._RL_grasp_pose__dont_move_block_active = False 
    env_grasp_pose._RL_grasp_pose__grasp_at_end_active = False

    env_placing_pose = RL_placing("get_in_pose_above1_8.zip", env_above_pose, "get_in_grasp_pose1_8.zip", env_grasp_pose)
    env_placing_pose._RL_placing__distance_reward_active = True
    env_placing_pose._RL_placing__start_close_to_goal_active = False
    env_placing_pose._RL_placing__grasp_at_end_active = False

    ## Start spinning nodes
    executor = MultiThreadedExecutor()
    executor.add_node(env_above_pose._ReinforcementLearnerEnvironment__node)
    executor.add_node(env_grasp_pose._RL_grasp_pose__node)
    executor.add_node(env_placing_pose._RL_placing__node)
    Thread(target = executor.spin).start()

    ##Learn position model:
    modelname = "get_in_place_pose_"
    number_of_models = 2
    learn_position(model_name=modelname, number_of_models=number_of_models, env = env_placing_pose)

    ##Test grasp success on position learner model:
    # model = DDPG.load("get_in_grasp_pose1_8.zip")
    # test_grasp_from_position_learner(model = model, env = env_grasp_pose)

    # ##Test model:
    # model = DDPG.load("get_in_grasp_pose1_8.zip")
    # test_model(model = model, env = env_grasp_pose)
    


if __name__ == '__main__':
    main()