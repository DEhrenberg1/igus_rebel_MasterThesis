import gymnasium as gym
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from gymnasium import spaces
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise  
#from controller import Supervisor
from stable_baselines3.common.logger import configure
import time
import pandas as pd
import threading
import math
##Ros2 Message Types
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from std_msgs.msg import Float64
from abc import ABC, abstractmethod

class RLGoalPosition(gym.Env, ABC):
    def __init__(self, goal_pos_reference, goal_pos_offset, sim_reset):
        # goal_pos_reference: The reference for the goal position (block1 or block2)
        # goal_pos_offset: The offset for the goal position (from block1 or block2, depending on reference)
        super().__init__
        ## Goal Position:
        self.goal_pos = [0.0, 0.0, 0.0] #xyz
        self.goal_pos_reference = goal_pos_reference #Possible values: "block1", "block2"
        self.goal_pos_offset = goal_pos_offset
        self.distance_gripper_goal = 2.0
        ## Constants
        self.neg_inf = np.finfo(np.float64).min #negative infinity
        self.pos_inf = np.finfo(np.float64).max #positive infinity
        self.max_steps = 100
        ## Observation variables
        self.arm_positions = [0.0,0.0,0.0,0.0,0.0,0.0]
        self.arm_velocities = [0.0,0.0,0.0,0.0,0.0,0.0]
        self.block1_pos = [0.0, 0.0, 0.0] #xyz
        self.block1_initial = self.block1_pos
        self.block2_pos = [0.0, 0.0, 0.0] #xyz
        self.pos_b1_set = False
        self.pos_b2_set = False
        self.pos_gripper_set = False
        self.safety_distance_ground = 0.03 #The safety distance from the pinch position to the ground in meters
        self.simulation_reset = sim_reset

        self.distance_reward_active = False #Set this to true to activate a small distance based reward

        ## OpenAI Stuff (observation and action space for RL)
        self.action_space = spaces.Box(low = -0.4, high = 0.4, shape = (4,), dtype=np.float64) #Velocity controller of all six joints
        self.observation_space = spaces.Dict(
            {
                "rebel_arm_position": spaces.Box(self.neg_inf, self.pos_inf, shape= (4,), dtype=np.float64),
                "rebel_arm_velocity": spaces.Box(-1.1, 1.1, shape = (4,), dtype=np.float64),
                "position_goal": spaces.Box(-3, 3, shape = (3,), dtype=np.float64),
                "position_gripper": spaces.Box(-3, 3, shape = (3,), dtype=np.float64),
                "distance_to_goal" : spaces.Box(0, self.pos_inf, shape = (1,), dtype=np.float64)
            }
        )
        self.current_steps = 0
        ## ROS2 Stuff
        self.initializeROS2Stuff()
    
    @abstractmethod
    def initializeROS2Stuff():
        #This method should start the needed ROS2 subscribers and publishers, as well as the ROS2 node.
        pass
    
    def limitedAction(self, action):
        lim_act = [action[0], action[1], action[2], 0.0, action[3], 0.0] #limited joint velocities
        joint_lim_max = [1.0,1.0,1.0,0.0,1.5,0.0]
        joint_lim_min = [-1.0,0.1,0.2,0.0,0.3,0.0]
        #Do not exceed the joint position limits specified in joint_lim_max and joint_lim_min:
        for i in range(6):
            if self.arm_positions[i] > joint_lim_max[i]:
                lim_act[i] = 0.0 if lim_act[i] > 0.0 else lim_act[i]
            if self.arm_positions[i] < joint_lim_min[i]:
                lim_act[i] = 0.0 if lim_act[i] < 0.0 else lim_act[i]
        
        return lim_act

    ## Rewriting step and reset function of OpenAIGym for our Use Case:
    def step(self, action):
        action = self.limitedAction(action)
        self.move_arm(action)
        self.current_steps += 1

        self.wait_for_move_execution() ##Different for Simulation and Real Robot

        observation = self.get_observation()
        reward = self.compute_reward()
        terminated, truncated = self.compute_if_done()
        info = {}
        return observation, reward, terminated, truncated, info
    
    @abstractmethod
    def wait_for_move_execution(self):
        pass

    def get_observation(self):
        self.compute_goal_position()
        self.distance_gripper_goal = self.compute_distance_gripper_goal_pos()
        
        rel_arm_pos = [self.arm_positions[0],self.arm_positions[1],self.arm_positions[2],self.arm_positions[4]]
        rel_arm_vel = [self.arm_velocities[0], self.arm_velocities[1], self.arm_velocities[2], self.arm_velocities[4]]
        observation = {"position_goal": self.goal_pos, "position_gripper": self.gripper_pos, "rebel_arm_position": rel_arm_pos, "rebel_arm_velocity": rel_arm_vel, "distance_to_goal": self.distance_gripper_goal}
        return observation
    
    def compute_reward(self):
        reward = 0
        if self.reached_goal_pose(0.02,0.02,0.02):
            reward = reward + 10
            print("Hurra!")
        if self.reached_goal_pose(0.01,0.01,0.01):
            reward = reward + 100
            print("Double Hurra!")
        
        if self.distance_reward_active:
            if self.distance_gripper_goal < 1.0:
                distance_reward = (10 ** (-1 * int(10*self.distance_gripper_goal))) * 5
                reward = reward + distance_reward
        
        #Gripper to close to ground:
        reward = reward - 0.5 if (self.gripper_pos[2] <= self.safety_distance_ground) else reward
        return reward
    
    def compute_if_done(self):
        terminated = False
        #End episode after max number of steps
        truncated = True if (self.current_steps>=self.max_steps) else False
        #End episode if gripper too close to the ground
        truncated = True if (self.gripper_pos[2] <= self.safety_distance_ground) else truncated

        if truncated:
            print(self.current_steps)
        return terminated, truncated

    @abstractmethod
    def reset (self, seed=None, options=None):
        pass

    ## The Functions below are used for ROS2-Communication:
    def joint_state_callback(self, joint_state):
        arm_name = joint_state.name
        self.arm_positions = []
        self.arm_velocities = []
        for i in range(6):
            ind = arm_name.index("joint" + str(i+1))
            self.arm_positions.append(joint_state.position[ind])
            self.arm_velocities.append(joint_state.velocity[ind])
        self.joint_state_set = True
    
    def pos_block1_callback(self, pos):
        self.block1_pos = [pos.x, pos.y, pos.z]
        self.pos_b1_set = True
    
    def pos_block2_callback(self, pos):
        self.block2_pos = [pos.x, pos.y, pos.z]
        self.pos_b2_set = True
    
    def pos_gripper_callback(self, pos):
        self.gripper_pos = [pos.x, pos.y, pos.z]
        self.pos_gripper_set = True

    @abstractmethod
    def move_arm(self, velocities):
        pass

    @abstractmethod        
    def move_gripper(self, msg):
        pass

    ##Function used to compute distance:

    def compute_distance_gripper_goal_pos(self):
        dist = math.sqrt((self.gripper_pos[0] - self.goal_pos[0])**2 + (self.gripper_pos[1] - self.goal_pos[1])**2 + (self.gripper_pos[2] - self.goal_pos[2])**2)
        return dist
    
    ##Function used to compute the reward from the observation:

    def reached_goal_pose(self, x_offset, y_offset, z_offset):
        #returns true if a plausible pose around the goal pose is reached (where plausible is defined by the offset)
        check_x = abs(self.goal_pos[0] - self.gripper_pos[0]) < x_offset
        check_y = abs(self.goal_pos[1] - self.gripper_pos[1]) < y_offset
        check_z = abs(self.goal_pos[2] - self.gripper_pos[2]) < z_offset

        return (check_x and check_y and check_z)
    
    ##Function used to compute the current goal position:

    def compute_goal_position(self):
        if self.goal_pos_reference == "block1":
            self.goal_pos = self.block1_pos
        elif self.goal_pos_reference == "block2":
            self.goal_pos = self.block2_pos
        else:
            print("Error: goal_pos_reference must be block1 or block2")
        
        self.goal_pos = [x + y for x, y in zip(self.goal_pos, self.goal_pos_offset)]

class RLUtilityClass:

    @staticmethod
    def learn_position(model_name, number_of_models, env):
        #Number of total_timesteps:
        high_act_noise_timesteps = 100000
        reduced_act_noise_timesteps_per_iteration = 50000
        low_act_noise_timeteps = 150000
        for j in range(number_of_models):
            # The noise object for DDPG
            action_noise = NormalActionNoise(mean=np.zeros(4,), sigma=np.ones(4,))
            model = DDPG("MultiInputPolicy", env, action_noise=action_noise, verbose=1, learning_rate = 0.001, tau = 0.001, learning_starts=10000, gamma = 0.99, batch_size=32  , buffer_size= 300000, gradient_steps= 4, train_freq = (1, "episode"))
            model.set_env(env)
            # set up logger
            path = model_name + str(j) + "log"
            new_logger = configure(path, ["stdout", "csv", "tensorboard"])  
            model.set_logger(new_logger)
            # learn
            model.learn(total_timesteps = high_act_noise_timesteps, log_interval=10)
            # save
            model.save(model_name + str(j))
            model.save_replay_buffer(model_name + str(j))

            #Do not train bad models further:
            # df = pd.read_csv(model_name + str(j) + "log" + "/progress.csv")
            # reward_column ="rollout/ep_rew_mean"
            # last_20_rewards = df[reward_column].tail(20)
            # average_reward = last_20_rewards.mean()

            # if average_reward < 1.7:
            #     continue

            #Learn with reduced action noise if model is promising:
            model = None
            for i in range(9):
                action_noise = NormalActionNoise(mean=np.zeros(4,), sigma= (0.9-i/10) * np.ones(4,))
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
            
            # Learn with low action noise
            action_noise = NormalActionNoise(mean=np.zeros(4,), sigma= 0.05 * np.ones(4,))
            model = DDPG.load(model_name + str(j) + "_" + "8.zip", learning_starts = 0, action_noise=action_noise)
            model.load_replay_buffer(model_name + str(j) + ".pkl")
            model.set_env(env)

            name = model_name + str(j) + "_" + "9"
            path = name + 'log'
            new_logger = configure(path, ["stdout", "csv", "tensorboard"]) 
            model.set_logger(new_logger)
            model.learn(total_timesteps=low_act_noise_timeteps, log_interval= 10)
            model.save(name)
            model.save_replay_buffer(name)

    @staticmethod
    def test_model(model, env):
        #model = DDPG.load("exact_position_learner_gui_true_0_8_trained_1_learned_gui_false")
        model.set_env(env)
        vec_env = model.get_env()
        obs = vec_env.reset()
        done = False
        while True:
            action, _states = model.predict(obs)
            obs, rewards, done, info = vec_env.step(action)
            
    @staticmethod
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
                    if env.block1_pos[2] > 0.1:
                        succeed = succeed + 1
                    obs = vec_env.reset()
                    reward = 0
                    done = True
            print("Success rate: " + str(succeed/trials) + ", Trials: " + str(trials))
    
    @staticmethod
    def grasp(model, env):
        model.set_env(env)
        vec_env = model.get_env()
        trials = 0
        succeed = 0
        while True:
            env.simulation_reset = True
            obs = vec_env.reset()
            trials = trials + 1
            #Get above grasp position
            done = False
            rewards = 0
            env.goal_pos_reference = "block1"
            env.goal_pos_offset = [0.0, 0.0, 0.05]
            env.simulation_reset = False #Make sure we stay in position above grasp position and do not reset
            while not done and rewards < 100:
                action, _states = model.predict(obs)
                obs, rewards, done, info = vec_env.step(action)
            #Get in grasp position
            env.goal_pos_offset = [0.0, 0.0, -0.025]
            rewards = 0
            done = False
            while not done and rewards < 100:
                action, _states = model.predict(obs)
                obs, rewards, done, info = vec_env.step(action)
            #Grasp and lift
            env.move_gripper(0.8)
            time.sleep(0.4)
            for _ in range(15):
                lim_act = env.limitedAction([0.0,-0.4,0.0,0.0])
                env.move_arm(lim_act)
                time.sleep(0.1)
            if env.block1_pos[2] > 0.1:
                        succeed = succeed + 1
            print("Success rate: " + str(succeed/trials) + ", Trials: " + str(trials))
            
