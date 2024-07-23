from goal_position_learner import RLGoalPosition
from goal_position_learner import RLUtilityClass
import rclpy
from rclpy.executors import MultiThreadedExecutor
import time
from threading import Thread
from stable_baselines3 import DDPG
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise  
import numpy as np

#ROS2 Message Types
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from std_msgs.msg import Float64

class RLGoalPosition_sim(RLGoalPosition):
    def __init__(self, goal_pos_reference, goal_pos_offset, sim_reset):
        super().__init__(goal_pos_reference, goal_pos_offset, sim_reset)

    def initializeROS2Stuff(self):
        self.node = rclpy.create_node('rl')
        self.node.create_subscription(JointState, '/joint_states', self.joint_state_callback, 1)
        self.node.create_subscription(Point, '/position/block1', self.pos_block1_callback, 1)
        self.node.create_subscription(Point, '/position/block2', self.pos_block2_callback, 1)
        self.node.create_subscription(Point, '/position/gripper', self.pos_gripper_callback, 1)
        self.arm_publisher = self.node.create_publisher(Float64MultiArray, '/joint_trajectory/command', 10)
        self.gripper_publisher = self.node.create_publisher(JointTrajectory, '/gripper_driver/command', 10)
        self.reset_publisher = self.node.create_publisher(Bool, '/reset', 10)
    
    def wait_for_move_execution(self):
        ## Wait until n simulation steps pass (where n is smaller when close to goal position)
        self.compute_goal_position()
        self.distance_gripper_goal = self.compute_distance_gripper_goal_pos()
        wait_time = 4
        if self.distance_gripper_goal <= 0.05:
            wait_time = 3
        if self.distance_gripper_goal <= 0.03:
            wait_time = 2
        if self.distance_gripper_goal <= 0.02:
            wait_time = 1
        for _ in range(wait_time):
            ##Condition for when the action got executed for one simulation step
            action_executed = False
            self.pos_b1_set = False
            self.pos_b2_set = False
            self.pos_gripper_set = False
            while not action_executed:
                action_executed = self.pos_b1_set and self.pos_b2_set and self.pos_gripper_set
    
    def reset(self, seed=None, options=None):
        print("Reset")
        self.current_steps = 0
        self.arm_positions = [0.0,0.0,0.0,0.0,0.0,0.0]
        self.arm_velocities = [0.0,0.0,0.0,0.0,0.0,0.0]
        old_b1 = self.block1_pos
        self.gripper_pos = [0.0, 0.0, 0.0] #xyz
        self.current_steps = 0

        ## Send reset-message to Webots Plugin (ObjectPositionPublisher)
        if self.simulation_reset:
            msg = Bool()
            msg.data = True
            self.reset_publisher.publish(msg)
            #Wait until world is reset and we know the new position of the blocks
            while(self.block1_pos == old_b1):
                time.sleep(0.01)
        
            time.sleep(0.1) #Wait until simulation is stable
        
        ## Return observations for RL
        observation = self.get_observation()
        info = {}
        return observation, info
    
    def move_arm(self, velocities):
        msg = Float64MultiArray()
        scaled_vel = [x*0.5 for x in velocities]
        msg.data = scaled_vel
        self.arm_publisher.publish(msg)

    def move_gripper(self, sent_msg):
        msg = JointTrajectory()
        msg.joint_names = ['left_finger_joint']
        point = JointTrajectoryPoint()
        point.positions = [sent_msg]
        msg.points.append(point)
        self.gripper_publisher.publish(msg)


def updater(node):
    while True:
        rclpy.spin(node)


def main(args = None):
    ## Initialisitation
    rclpy.init(args=args)
    env = RLGoalPosition_sim("block1", [0.0, 0.0, 0.0], True)
    env.distance_reward_active = True


    ## Start spinning nodes
    executor = MultiThreadedExecutor()
    executor.add_node(env.node)
    Thread(target = executor.spin).start()

    ## Learn position model:
    model_name = "get_in_goal_pose_v11(no_gui)_0_"
    action_noise = NormalActionNoise(mean=np.zeros(4,), sigma= 0.05 * np.ones(4,))
    model = DDPG.load(model_name + "8.zip", learning_starts = 0, action_noise=action_noise)
    model.load_replay_buffer(model_name + "8.pkl")
    model.set_env(env)

    name = model_name + "9"
    path = name + 'log'
    new_logger = configure(path, ["stdout", "csv", "tensorboard"]) 
    model.set_logger(new_logger)
    model.learn(total_timesteps=100000, log_interval= 10)
    model.save(name)
    model.save_replay_buffer(name)

    ##Test grasp success on position learner model:
    # model = DDPG.load("get_in_grasp_pose1_8.zip")
    # test_grasp_from_position_learner(model = model, env = env_grasp_pose)

    #Test model:
    # model = DDPG.load("get_in_goal_pose_v2_0_9")
    # RLUtilityClass.grasp(model, env1)

    # ##Test model:
    # model1 = DDPG.load("get_in_goal_pose_0_8.zip")
    # model1.set_env(env1)
    # model2 = DDPG.load("get_in_goal_pose_0_8.zip")
    # model2.set_env(env2)
    # model3 = DDPG.load("get_in_goal_pose_0_8.zip")
    # model3.set_env(env3)
    # model4 = DDPG.load("get_in_goal_pose_0_8.zip")
    # model4.set_env(env4)
    # vec_env1 = model1.get_env()
    # vec_env2 = model2.get_env()
    # vec_env3 = model3.get_env()
    # vec_env4 = model4.get_env()
    # while True:
        
    #     env1.simulation_reset = True
    #     obs = vec_env1.reset()
    #     env1.simulation_reset = False
    #     done = False
    #     rewards = 0
    #     while not done and rewards < 15:
    #         action, _states = model1.predict(obs)
    #         obs, rewards, done, info = vec_env1.step(action)
        
    #     obs = vec_env2.reset()
    #     done = False
    #     rewards = 0
    #     while not done and rewards < 100:
    #         action, _states = model2.predict(obs)
    #         obs, rewards, done, info = vec_env2.step(action)

    #     obs = vec_env3.reset()
    #     done = False
    #     rewards = 0
    #     while not done and rewards < 15:
    #         action, _states = model3.predict(obs)
    #         obs, rewards, done, info = vec_env3.step(action)
        
    #     obs = vec_env4.reset()
    #     done = False
    #     rewards = 0
    #     while not done and rewards < 15:
    #         action, _states = model4.predict(obs)
    #         obs, rewards, done, info = vec_env4.step(action)
    


if __name__ == '__main__':
    main()