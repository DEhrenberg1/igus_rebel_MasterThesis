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

class RL_goal_position(gym.Env):
    def __init__(self, goal_pos_reference, goal_pos_offset, sim_reset):
        # goal_pos_reference: The reference for the goal position (block1 or block2)
        # goal_pos_offset: The offset for the goal position (from block1 or block2, depending on reference)
        super().__init__
        ## Goal Position:
        self.__goal_pos = [0.0, 0.0, 0.0] #xyz
        self.__goal_pos_reference = goal_pos_reference #Possible values: "block1", "block2"
        self.__goal_pos_offset = goal_pos_offset
        self.__distance_gripper_goal = 2.0
        ## Constants
        self.__neg_inf = np.finfo(np.float64).min #negative infinity
        self.__pos_inf = np.finfo(np.float64).max #positive infinity
        self.__max_steps = 50
        ## Observation variables
        self.__arm_positions = [0.0,0.0,0.0,0.0,0.0,0.0]
        self.__arm_velocities = [0.0,0.0,0.0,0.0,0.0,0.0]
        self.__block1_pos = [0.0, 0.0, 0.0] #xyz
        self.__block1_initial = self.__block1_pos
        self.__block2_pos = [0.0, 0.0, 0.0] #xyz
        self.__pos_b1_set = False
        self.__pos_b2_set = False
        self.__pos_gripper_set = False
        self.__safety_distance_ground = 0.03 #The safety distance from the pinch position to the ground in meters
        self.simulation_reset = sim_reset

        self.__distance_reward_active = False #Set this to true to activate a small distance based reward

        ## OpenAI Stuff (observation and action space for RL)
        self.action_space = spaces.Box(low = -0.4, high = 0.4, shape = (4,), dtype=np.float64) #Velocity controller of all six joints
        self.observation_space = spaces.Dict(
            {
                "rebel_arm_position": spaces.Box(self.__neg_inf, self.__pos_inf, shape= (4,), dtype=np.float64),
                "rebel_arm_velocity": spaces.Box(-1.1, 1.1, shape = (4,), dtype=np.float64),
                "position_goal": spaces.Box(-3, 3, shape = (3,), dtype=np.float64),
                "position_gripper": spaces.Box(-3, 3, shape = (3,), dtype=np.float64),
                "distance_to_goal" : spaces.Box(0, self.__pos_inf, shape = (1,), dtype=np.float64)
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
        ## Wait until n simulation steps pass (where n is smaller when close to goal position)
        self.compute_goal_position()
        self.__distance_gripper_goal = self.compute_distance_gripper_goal_pos()
        wait_time = 4
        if self.__distance_gripper_goal <= 0.05:
            wait_time = 3
        if self.__distance_gripper_goal <= 0.03:
            wait_time = 2
        if self.__distance_gripper_goal <= 0.02:
            wait_time = 1
        for _ in range(wait_time):
            ##Condition for when the action got executed for one simulation step
            action_executed = False
            self.__pos_b1_set = False
            self.__pos_b2_set = False
            self.__pos_gripper_set = False
            while not action_executed:
                action_executed = self.__pos_b1_set and self.__pos_b2_set and self.__pos_gripper_set

        self.compute_goal_position()
        self.__distance_gripper_goal = self.compute_distance_gripper_goal_pos()
        
        rel_arm_pos = [self.__arm_positions[0],self.__arm_positions[1],self.__arm_positions[2],self.__arm_positions[4]]
        rel_arm_vel = [self.__arm_velocities[0], self.__arm_velocities[1], self.__arm_velocities[2], self.__arm_velocities[4]]
        observation = {"position_goal": self.__goal_pos, "position_gripper": self.__gripper_pos, "rebel_arm_position": rel_arm_pos, "rebel_arm_velocity": rel_arm_vel, "distance_to_goal": self.__distance_gripper_goal}

        reward = 0
        if self.reached_goal_pose(0.02,0.02,0.02):
            reward = reward + 10
            print("Hurra!")
        if self.reached_goal_pose(0.01,0.01,0.01):
            reward = reward + 100
            print("Double Hurra!")
        
        if self.__distance_reward_active:
            if self.__distance_gripper_goal < 1.0:
                distance_reward = (10 ** (-1 * int(10*self.__distance_gripper_goal))) * 5
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
        self.__arm_positions = [0.0,0.0,0.0,0.0,0.0,0.0]
        self.__arm_velocities = [0.0,0.0,0.0,0.0,0.0,0.0]
        old_b1 = self.__block1_pos
        self.__gripper_pos = [0.0, 0.0, 0.0] #xyz
        self.__current_steps = 0

        ## Send reset-message to Webots Plugin (ObjectPositionPublisher)
        if self.simulation_reset:
            msg = Bool()
            msg.data = True
            self.reset_publisher.publish(msg)
            #Wait until world is reset and we know the new position of the blocks
            while(self.__block1_pos == old_b1):
                time.sleep(0.01)
        
            time.sleep(0.1) #Wait until simulation is stable
        
        ## Return observations for RL
        self.compute_goal_position()
        self.__distance_gripper_goal = self.compute_distance_gripper_goal_pos()

        rel_arm_pos = [self.__arm_positions[0],self.__arm_positions[1],self.__arm_positions[2],self.__arm_positions[4]]
        rel_arm_vel = [self.__arm_velocities[0], self.__arm_velocities[1], self.__arm_velocities[2], self.__arm_velocities[4]]
        observation = {"position_goal": self.__goal_pos, "position_gripper": self.__gripper_pos, "rebel_arm_position": rel_arm_pos, "rebel_arm_velocity": rel_arm_vel, "distance_to_goal": self.__distance_gripper_goal}
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
        self.__block1_pos = [pos.x, pos.y, pos.z]
        self.__pos_b1_set = True
    
    def __pos_block2_callback(self, pos):
        self.__block2_pos = [pos.x, pos.y, pos.z]
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

    ##Function used to compute distance:

    def compute_distance_gripper_goal_pos(self):
        dist = math.sqrt((self.__gripper_pos[0] - self.__goal_pos[0])**2 + (self.__gripper_pos[1] - self.__goal_pos[1])**2 + (self.__gripper_pos[2] - self.__goal_pos[2])**2)
        return dist
    
    ##Function used to compute the reward from the observation:

    def reached_goal_pose(self, x_offset, y_offset, z_offset):
        #returns true if a plausible pose around the goal pose is reached (where plausible is defined by the offset)
        check_x = abs(self.__goal_pos[0] - self.__gripper_pos[0]) < x_offset
        check_y = abs(self.__goal_pos[1] - self.__gripper_pos[1]) < y_offset
        check_z = abs(self.__goal_pos[2] - self.__gripper_pos[2]) < z_offset

        return (check_x and check_y and check_z)
    
    ##Function used to compute the current goal position:

    def compute_goal_position(self):
        if self.__goal_pos_reference == "block1":
            self.__goal_pos = self.__block1_pos
        elif self.__goal_pos_reference == "block2":
            self.__goal_pos = self.__block2_pos
        else:
            print("Error: goal_pos_reference must be block1 or block2")
        
        self.__goal_pos = [x + y for x, y in zip(self.__goal_pos, self.__goal_pos_offset)]

    

def learn_position(model_name, number_of_models, env):
    #Number of total_timesteps:
    high_act_noise_timesteps = 500000
    reduced_act_noise_timesteps_per_iteration = 30000
    low_act_noise_timeteps = 150000
    # The noise object for DDPG
    action_noise = NormalActionNoise(mean=np.zeros(4,), sigma=np.ones(4,))
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
                if env._RL_grasp_pose__block1_pos[2] > 0.1:
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
    env1 = RL_goal_position("block1", [0.0, 0.0, 0.05], True)
    env1._RL_goal_position__distance_reward_active = True

    env2 = RL_goal_position("block1", [0.0, 0.0, 0.0], False)
    env2._RL_goal_position__distance_reward_active = True

    env3 = RL_goal_position("block2", [0.0, 0.1, 0.05], False)
    env3._RL_goal_position__distance_reward_active = True

    env4 = RL_goal_position("block2", [0.0, 0.0, 0.05], False)
    env4._RL_goal_position__distance_reward_active = True

    ## Start spinning nodes
    executor = MultiThreadedExecutor()
    executor.add_node(env1._RL_goal_position__node)
    executor.add_node(env2._RL_goal_position__node)
    executor.add_node(env3._RL_goal_position__node)
    executor.add_node(env4._RL_goal_position__node)
    Thread(target = executor.spin).start()

    ## Learn position model:
    # modelname = "get_in_goal_pose_"
    # number_of_models = 2
    # learn_position(model_name=modelname, number_of_models=number_of_models, env = env)

    ##Test grasp success on position learner model:
    # model = DDPG.load("get_in_grasp_pose1_8.zip")
    # test_grasp_from_position_learner(model = model, env = env_grasp_pose)

    # ##Test model:
    model1 = DDPG.load("get_in_goal_pose_0_8.zip")
    model1.set_env(env1)
    model2 = DDPG.load("get_in_goal_pose_0_8.zip")
    model2.set_env(env2)
    model3 = DDPG.load("get_in_goal_pose_0_8.zip")
    model3.set_env(env3)
    model4 = DDPG.load("get_in_goal_pose_0_8.zip")
    model4.set_env(env4)
    vec_env1 = model1.get_env()
    vec_env2 = model2.get_env()
    vec_env3 = model3.get_env()
    vec_env4 = model4.get_env()
    while True:
        
        env1.simulation_reset = True
        obs = vec_env1.reset()
        env1.simulation_reset = False
        done = False
        rewards = 0
        while not done and rewards < 15:
            action, _states = model1.predict(obs)
            obs, rewards, done, info = vec_env1.step(action)
        
        obs = vec_env2.reset()
        done = False
        rewards = 0
        while not done and rewards < 100:
            action, _states = model2.predict(obs)
            obs, rewards, done, info = vec_env2.step(action)

        obs = vec_env3.reset()
        done = False
        rewards = 0
        while not done and rewards < 15:
            action, _states = model3.predict(obs)
            obs, rewards, done, info = vec_env3.step(action)
        
        obs = vec_env4.reset()
        done = False
        rewards = 0
        while not done and rewards < 15:
            action, _states = model4.predict(obs)
            obs, rewards, done, info = vec_env4.step(action)
    


if __name__ == '__main__':
    main()