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
from threading import Thread
##Ros2 Message Types
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from std_msgs.msg import Float64


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
        self.__sim_time = 0.0
        self.__sim_time_set = False
        self.__start_time = 0.0
        self.__reached_grasp_pos = False
        self.__grasped = False
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
        self.__node.create_subscription(JointState, '/joint_states', self.__joint_state_callback, 1)
        self.__node.create_subscription(Point, '/position/block1', self.__pos_block1_callback, 1)
        self.__node.create_subscription(Point, '/position/block2', self.__pos_block2_callback, 1)
        self.__node.create_subscription(Point, '/position/gripper', self.__pos_gripper_callback, 1)
        self.__node.create_subscription(Point, '/distance', self.__distance_callback, 1)
        self.__node.create_subscription(Float64, '/sim_time', self.__sim_time_callback, 1)
        #self.__arm_publisher = self.__node.create_publisher(Float64MultiArray, '/rebel_arm_controller/commands', 10)
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
        for _ in range(4):
            ##Condition for when the action is completed
            action_executed = False
            self.__pos_b1_set = False
            self.__pos_b2_set = False
            #self.__joint_state_set = False
            self.__pos_gripper_set = False
            self.__distance_set = False
            self.__sim_time_set = False
            #i = datetime.datetime.now()
            while not action_executed:
                action_executed = self.__pos_b1_set and self.__pos_b2_set and self.__joint_state_set and self.__pos_gripper_set and self.__distance_set and self.__sim_time_set

        pos_block1 = [self.__block1_x, self.__block1_y, self.__block1_z]
        pos_gripper = [self.__gripper_x, self.__gripper_y, self.__gripper_z]
        rel_arm_pos = [self.__arm_positions[0],self.__arm_positions[1],self.__arm_positions[2],self.__arm_positions[4]]
        rel_arm_vel = [self.__arm_velocities[0], self.__arm_velocities[1], self.__arm_velocities[2], self.__arm_velocities[4]]

        observation = {"position_block_1": pos_block1, "position_gripper": pos_gripper, "rebel_arm_position": rel_arm_pos, "rebel_arm_velocity": rel_arm_vel, "distance_to_block": self.__distance_gripper_b1}
        #observation = {"distance_to_block": self.__distance_gripper_b1}

        reward = 0

        if self.reached_grasp_pose(0.02,0.02,0.02):
            reward = reward + 10
            print("Hurra!")
        
        terminated = False
        truncated = True if (self.__current_steps>=self.__max_steps) else False
        truncated = True if (self.__gripper_z <= 0.004) else truncated
        if truncated:
            print(self.__current_steps)
        info = {}
        return observation, reward, terminated, truncated, info
    
    def reset (self, seed=None, options=None):
        super().reset(seed=seed)
        print("simulation will be reset")

        self.__arm_positions = [0.0,0.0,0.0,0.0,0.0,0.0]
        self.__arm_velocities = [0.0,0.0,0.0,0.0,0.0,0.0]
        old_b1x = self.__block1_x
        old_b1y = self.__block1_y
        old_b1z = self.__block1_z
        old_b2x = self.__block2_x
        old_b2y = self.__block2_y
        old_b2z = self.__block2_z
        self.__gripper_x = 0.0
        self.__gripper_y = 0.0
        self.__gripper_z = 0.0
        self.__distance_gripper_b1 = 2.0
        self.__distance_gripper_b2 = self.__pos_inf
        self.__distance_b1_b2 = self.__pos_inf
        self.__current_steps = 0
        self.__reached_grasp_pos = False
        self.__grasped = False

        msg = Bool()
        msg.data = True
        self.reset_publisher.publish(msg)

        #Wait until world is reset and we know the new position of the blocks
        while(self.__block1_x == old_b1x and self.__block1_y == old_b1y):
            pass

        time.sleep(0.1) #Wait until simulation is stable

        pos_block1 = [self.__block1_x, self.__block1_y, self.__block1_z]
        pos_gripper = [self.__gripper_x, self.__gripper_y, self.__gripper_z]

        rel_arm_pos = [self.__arm_positions[0],self.__arm_positions[1],self.__arm_positions[2],self.__arm_positions[4]]
        rel_arm_vel = [self.__arm_velocities[0], self.__arm_velocities[1], self.__arm_velocities[2], self.__arm_velocities[4]]
        observation = {"position_block_1": pos_block1, "position_gripper": pos_gripper, "rebel_arm_position": rel_arm_pos, "rebel_arm_velocity": rel_arm_vel, "distance_to_block": [self.__distance_gripper_b1]}
        info = {}
        return observation, info
    
    #Not sure if these functions are needed.
    #def render(self):
    #    pass

    #def close(self):
    #    pass

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
        self.__gripper_x = pos.x
        self.__gripper_y = pos.y
        self.__gripper_z = pos.z
        self.__pos_gripper_set = True

    def __distance_callback(self, pos):
        self.__distance_b1_b2 = pos.x
        self.__distance_gripper_b1 = pos.y
        self.__distance_gripper_b2 = pos.z
        self.__distance_set = True
    
    def __sim_time_callback(self, msg):
        self.__sim_time = msg.data
        self.__sim_time_set = True
    
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
    # The noise object for DDPG
    action_noise = NormalActionNoise(mean=np.zeros(4,), sigma=1.0 * np.ones(4,))
    # action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(4,), sigma=1.0 * np.ones(4,), theta = 0.01)

    
    model = DDPG("MultiInputPolicy", env, action_noise=action_noise, verbose=1, learning_rate = 0.001, tau = 0.001, learning_starts=50000, gamma = 0.99, batch_size=32  , buffer_size= 300000, gradient_steps= 4, train_freq = (1, "episode"))
    #model = DDPG.load("exact_position_learner", learning_starts = 0, action_noise = action_noise, gradient_steps = 5)
    model.set_env(env)
    #model = PPO("MultiInputPolicy", env=env, batch_size=3,n_epochs=2,n_steps=450)
    tmp_path = "/tmp/sb3_log/"
    # set up logger
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])  
    model.set_logger(new_logger)
    model.learn(total_timesteps = 300000, log_interval=10)
    #test 6 war 0.03 bei allen
    model.save("exact_position_learner_gui_true_0")

    #learn with reduced action noise
    for i in range(9):
        action_noise = NormalActionNoise(mean=np.zeros(4,), sigma=(0.9-i/10) * np.ones(4,))
        if i == 0:
            model = DDPG.load("exact_position_learner_gui_true_0", learning_starts = 0, action_noise=action_noise)
        else:
            del model
            model = DDPG.load("exact_position_learner_gui_true_0_" + str(i-1), learning_starts = 0, action_noise=action_noise)
        model.set_env(env)
        model.learn(total_timesteps=50000, log_interval= 10)
        model.save("exact_position_learner_gui_true_0_" + str(i))
    
    # model.learn(total_timesteps=100000, log_interval=1)
    # model.save("exact_position_learner_1_9")

    # model = DDPG.load("exact_position_learner_gui_false_0_8")
    # model.set_env(env)
    # vec_env = model.get_env()
    # obs = vec_env.reset()
    # done = False
    # while True:
    #     action, _states = model.predict(obs)
    #     obs, rewards, done, info = vec_env.step(action)


    # model = DDPG.load("exact_position_learner_0_9")
    # model.set_env(env)
    # vec_env = model.get_env()
    # obs = vec_env.reset()
    # done = False
    # reward = 0
    # succeed = 0
    # failed = 0
    # trials = 0
    # while True:
    #     done = False
    #     trials = trials + 1
    #     while not done:
    #         action, _states = model.predict(obs)
    #         obs, rewards, done, info = vec_env.step(action)
    #         reward = reward + rewards
    #         if reward > 10:
    #             env.move_gripper(0.8)
    #             time.sleep(0.4)
    #             for _ in range(15):
    #                 lim_act = env.limitedAction([0.0,-0.4,0.0,0.0])
    #                 env.move_arm(lim_act)
    #                 env.move_gripper(0.8)
    #                 time.sleep(0.1)
    #             if env._ReinforcementLearnerEnvironment__block1_z > 0.05:
    #                 succeed = succeed + 1
    #             obs = vec_env.reset()
    #             reward = 0
    #             done = True
    #     print("Success rate: " + str(succeed/trials) + ", Trials: " + str(trials))


        # if reward > 10:
        #     succeed = succeed + 1
        # else:
        #     failed = failed + 1
        # success_rate = succeed / (succeed + failed)
        # print("Success rate: " + str(success_rate))
        # print("Number of tries: " + str((succeed + failed)))
        # reward = 0
        # done = False

if __name__ == '__main__':
    main()