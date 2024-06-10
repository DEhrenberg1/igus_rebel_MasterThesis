import gymnasium as gym
import numpy as np
import rclpy
from gymnasium import spaces
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
#from controller import Supervisor
from stable_baselines3.common.logger import configure
import time
import pandas as pd
from threading import Thread
##Ros2 Message Types
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from std_msgs.msg import Float64
import math


class ReinforcementLearnerEnvironment(gym.Env):
    def __init__(self):
        super().__init__
        ## Constants
        self.__neg_inf = np.finfo(np.float64).min #negative infinity
        self.__pos_inf = np.finfo(np.float64).max #positive infinity
        self.__max_steps = 50
        ## Observation variables
        self.__arm_positions = [0.0,0.0,0.0,0.0,0.0,0.0]
        self.__arm_velocities = [0.0,0.0,0.0,0.0,0.0,0.0]
        # self.__block1_x = 0.45
        # self.__block1_y = 0.05
        # self.__block1_z = 0.0275
        self.__block1_x = 0.0
        self.__block1_y = 0.0
        self.__block1_z = 0.0
        self.__block1_initial = [self.__block1_x, self.__block1_y, self.__block1_z]
        self.__gripper_x = 0.0
        self.__gripper_y = 0.0
        self.__gripper_z = 0.0
        self.__distance_gripper_b1 = 2.0

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
                "position_block_1": spaces.Box(-3, 3, shape = (3,), dtype=np.float64),
                "position_gripper": spaces.Box(-3, 3, shape = (3,), dtype=np.float64),
                "distance_to_block" : spaces.Box(0, self.__pos_inf, shape = (1,), dtype=np.float64)
            }
        )
        self.__current_steps = 0
        ## ROS2 Stuff
        self.__node = rclpy.create_node('reinforcement_learner')
        self.__node.create_subscription(JointState, '/real/joint_states', self.__joint_state_callback, 1)
        self.__node.create_subscription(Point, '/real/position/block1', self.__pos_block1_callback, 1)
        self.__node.create_subscription(Point, '/real/position/gripper', self.__pos_gripper_callback, 1)
        self.__node.create_subscription(Point, '/real/distance', self.__distance_callback, 1)
        self.__arm_publisher = self.__node.create_publisher(JointTrajectory, '/real/joint_trajectory_controller/joint_trajectory', 10)
        #self.__gripper_publisher = self.__node.create_publisher(JointTrajectory, '/gripper_driver/command', 10)
    
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
        
        #Wait until move is executed.
        time.sleep(0.4)

        pos_block1 = [self.__block1_x, self.__block1_y, self.__block1_z]
        pos_gripper = [self.__gripper_x, self.__gripper_y, self.__gripper_z]
        rel_arm_pos = [self.__arm_positions[0],self.__arm_positions[1],self.__arm_positions[2],self.__arm_positions[4]]
        rel_arm_vel = [self.__arm_velocities[0], self.__arm_velocities[1], self.__arm_velocities[2], self.__arm_velocities[4]]
        self.__distance_gripper_b1 = self.compute_distance_gripper_b1()
        
        #print(pos_block1)
        print (self.__distance_gripper_b1)
        observation = {"position_block_1": pos_block1, "position_gripper": pos_gripper, "rebel_arm_position": rel_arm_pos, "rebel_arm_velocity": rel_arm_vel, "distance_to_block": self.__distance_gripper_b1}

        reward = 0

        if self.reached_grasp_pose(0.02,0.02,0.02):
            reward = reward + 10
            print("Hurra!")

        if self.__distance_reward_active:
            if self.__distance_gripper_b1 < 1:
                distance_reward = 10 ** (-1 * int(10*self.__distance_gripper_b1))
                reward = reward + distance_reward
        
        terminated = False
        truncated = True if (self.__current_steps>=self.__max_steps) else False
        truncated = True if (self.__gripper_z <= 0.03) else truncated
        
        #reset if block was moved too much
        if self.__dont_move_block_active:
            for i in range(3):
                if abs(pos_block1[i] - self.__block1_initial[i]) > 0.005:
                    truncated = True
        if self.__grasp_at_end_active:
            if not truncated:
                if self.reached_grasp_pose(0.01,0.01,0.02):
                    self.move_gripper(0.8)
                    time.sleep(0.4)
                    lim_act = self.limitedAction([0.0,-0.4,0.0,0.0])
                    for _ in range(2):
                        self.move_arm(lim_act)
                        time.sleep(0.1)
                    if self.__block1_z > 0.035:
                        reward = 300
                        print("Lifted!")
                    terminated = True

        if truncated:
            print(self.__current_steps)
        info = {}
        return observation, reward, terminated, truncated, info
    
    def simulation_reset (self):
        self.__arm_positions = [0.0,0.0,0.0,0.0,0.0,0.0]
        self.__arm_velocities = [0.0,0.0,0.0,0.0,0.0,0.0]
        #self.__block1_x = 0.0
        #self.__block1_y = 0.0
        #self.__block1_z = 0.0
        self.__gripper_x = 0.0
        self.__gripper_y = 0.0
        self.__gripper_z = 0.0
        self.__distance_gripper_b1 = 2.0
        self.__current_steps = 0

        # Open Gripper
        # Move Arm to some random new position
        # Wait until Arm is at new position

        time.sleep(2) #Wait
    
    def reset (self, seed=None, options=None):
        super().reset(seed=seed)
        print("simulation will be reset")
        self.simulation_reset()

        if self.__start_close_to_goal_active:
            while(self.__distance_gripper_b1 > 0.35):
                print("reset cause too far away")
                self.simulation_reset()


        self.__block1_initial = [self.__block1_x, self.__block1_y, self.__block1_z]
        pos_block1 = [self.__block1_x, self.__block1_y, self.__block1_z]
        pos_gripper = [self.__gripper_x, self.__gripper_y, self.__gripper_z]

        rel_arm_pos = [self.__arm_positions[0],self.__arm_positions[1],self.__arm_positions[2],self.__arm_positions[4]]
        rel_arm_vel = [self.__arm_velocities[0], self.__arm_velocities[1], self.__arm_velocities[2], self.__arm_velocities[4]]
        self.__distance_gripper_b1 = self.compute_distance_gripper_b1()

        observation = {"position_block_1": pos_block1, "position_gripper": pos_gripper, "rebel_arm_position": rel_arm_pos, "rebel_arm_velocity": rel_arm_vel, "distance_to_block": [self.__distance_gripper_b1]}
        info = {}
        return observation, info
    

    ## The Functions below are used for ROS2-Communication:
    def __joint_state_callback(self, joint_state):
        arm_name = joint_state.name
        if len(joint_state.velocity) == 6 and len(joint_state.position) == 6:
            self.__arm_positions = []
            self.__arm_velocities = []
            for i in range(6):
                ind = arm_name.index("joint" + str(i+1))
                self.__arm_positions.append(joint_state.position[ind])
                self.__arm_velocities.append(joint_state.velocity[ind])
    
    def __pos_block1_callback(self, pos):
        self.__block1_x = pos.x
        self.__block1_y = pos.y
        self.__block1_z = pos.z
    
    def __pos_gripper_callback(self, pos):
        self.__gripper_x = pos.x
        self.__gripper_y = pos.y
        self.__gripper_z = pos.z

    def __distance_callback(self, pos):
        self.__distance_b1_b2 = pos.x
        self.__distance_gripper_b1 = pos.y
        self.__distance_gripper_b2 = pos.z
    
    def move_arm(self, velocities):
        scaled_vel = [x * 12.5 for x in velocities] #Scale up velocity for real robot
        msg = JointTrajectory()
        msg.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        point = JointTrajectoryPoint()
        point.positions = [0.0,0.0,0.0,0.0,0.0,0.0]
        point.velocities = scaled_vel
        msg.points.append(point)
        self.__arm_publisher.publish(msg)
    
    # def move_gripper(self, pos):
    #     msg = JointTrajectory()
    #     msg.joint_names = ['left_finger_joint']
    #     point = JointTrajectoryPoint()
    #     point.positions = [pos]
    #     msg.points.append(point)
    #     self.__gripper_publisher.publish(msg)
    
    ##Function to compute the distance:

    def compute_distance_gripper_b1(self):
        dist = math.sqrt((self.__gripper_x - self.__block1_x)**2 + (self.__gripper_y - self.__block1_y)**2 + (self.__gripper_z - self.__block1_z)**2)
        return dist

    ##Functions used to compute the reward from the observation:

    def reached_grasp_pose(self, x_offset, y_offset, z_offset):
        #returns true if a plausible grasp pose is reached (where plausible is defined by the offset)
        check_x = abs(self.__block1_x - self.__gripper_x) < x_offset
        check_y = abs(self.__block1_y - self.__gripper_y) < y_offset
        check_z = abs(self.__block1_z - self.__gripper_z) < z_offset

        return (check_x and check_y and check_z)

def learn_position(model_name, number_of_models, env):
    #Number of total_timesteps:
    high_act_noise_timesteps = 300000
    reduced_act_noise_timesteps_per_iteration = 50000
    low_act_noise_timeteps = 250000
    # The noise object for DDPG
    action_noise = NormalActionNoise(mean=np.zeros(4,), sigma=1.0 * np.ones(4,))
    model = DDPG("MultiInputPolicy", env, action_noise=action_noise, verbose=1, learning_rate = 0.001, tau = 0.001, learning_starts=50000, gamma = 0.99, batch_size=32  , buffer_size= 300000, gradient_steps= 4, train_freq = (1, "episode"))
    model.set_env(env)
    for j in range(number_of_models):
        # set up logger
        path = model_name + str(j)
        new_logger = configure(path, ["stdout", "csv", "tensorboard"])  
        model.set_logger(new_logger)
        # learn
        model.learn(total_timesteps = high_act_noise_timesteps, log_interval=1)
        # save
        model.save(model_name + str(j))
        model.save_replay_buffer(model_name + str(j))

        #Do not train bad models further:
        df = pd.read_csv(model_name + str(j) +"/progress.csv")
        reward_column ="rollout/ep_rew_mean"
        last_20_rewards = df[reward_column].tail(20)
        average_reward = last_20_rewards.mean()

        if average_reward < 1.7:
            continue

        #Learn with reduced action noise if model is promising:
        model = None
        for i in range(9):
            action_noise = NormalActionNoise(mean=np.zeros(4,), sigma=(0.9-i/10) * np.ones(4,))
            if i == 0:
                model = DDPG.load(model_name + str(j) + ".zip", learning_starts = 0, action_noise=action_noise)
                model.load_replay_buffer(model_name + str(j) + ".pkl")
            else:
                model = DDPG.load(model_name + str(j) + "_" + str(i-1)+".zip", learning_starts = 0, action_noise=action_noise)
                model.load_replay_buffer(model_name + str(j) + "_" + str(i-1) + ".pkl")
            model.set_env(env)
            # set up logger
            name = model_name + str(j) + "_"  + str(i)
            path = name
            new_logger = configure(path, ["stdout", "csv", "tensorboard"])  
            model.set_logger(new_logger)
            model.learn(total_timesteps=reduced_act_noise_timesteps_per_iteration, log_interval= 10)
            model.save(name)
            model.save_replay_buffer(name)
        
        model.learn(total_timesteps=low_act_noise_timeteps, log_interval=1)
        model.save("exact_position_learner_1_9")

def test_model(model, env):
    model = DDPG.load("exact_position_learner_gui_true_0_8_trained_1_learned_gui_false")
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
            if reward > 10:
                env.move_gripper(0.8)
                time.sleep(0.4)
                for _ in range(15):
                    lim_act = env.limitedAction([0.0,-0.4,0.0,0.0])
                    env.move_arm(lim_act)
                    time.sleep(0.1)
                if env._ReinforcementLearnerEnvironment__block1_z > 0.05:
                    succeed = succeed + 1
                obs = vec_env.reset()
                reward = 0
                done = True
        print("Success rate: " + str(succeed/trials) + ", Trials: " + str(trials))

def updater(node):
    while True:
        rclpy.spin(node)

def main(args = None):
    ## Initialisitation
    rclpy.init(args=args)
    env = ReinforcementLearnerEnvironment()
    Thread(target = updater, args = [env._ReinforcementLearnerEnvironment__node]).start() #Spin Node to update values
    env._ReinforcementLearnerEnvironment__distance_reward_active = False
    env._ReinforcementLearnerEnvironment__start_close_to_goal_active = False
    ##Learn position model:
    # modelname = "distance_reward_active_"
    # number_of_models = 2
    # learn_position(model_name=modelname, number_of_models=number_of_models, env = env)

    ##Test grasp success on position learner model:
    # model = DDPG.load("exact_position_learner_gui_true_0_8_trained_1")
    # test_grasp_from_position_learner(model = model, env = env)

    ##Test model:
    # model = DDPG.load("exact_position_learner_gui_true_0_8_trained_1_learned_gui_false")
    # test_model(model = model, env = env)

    #model = DDPG.load("exact_position_learner_gui_true_0_8.zip")
    model = DDPG.load("minimize_distance_1_8.zip")
    model.set_env(env)
    vec_env = model.get_env()
    obs = vec_env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = vec_env.step(action)

    #env.move_arm([0.0,-0.4,0.0,0.0,0.0,0.0])
    


if __name__ == '__main__':
    main()