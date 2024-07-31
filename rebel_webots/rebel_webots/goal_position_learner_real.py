from goal_position_learner import RLGoalPosition
from goal_position_learner import RLUtilityClass
import rclpy
from rclpy.executors import MultiThreadedExecutor
import time
from threading import Thread
import numpy as np
from stable_baselines3 import DDPG

#ROS2 Message Types
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from std_msgs.msg import Float64
from std_msgs.msg import String

class RLGoalPosition_real(RLGoalPosition):
    def __init__(self, goal_pos_reference, goal_pos_offset, sim_reset):
        super().__init__(goal_pos_reference, goal_pos_offset, sim_reset)

    def initializeROS2Stuff(self):
        self.node = rclpy.create_node('goal_position_learner_real')
        self.node.create_subscription(JointState, '/real/filtered_joint_states', self.joint_state_callback, 1)
        self.node.create_subscription(Point, '/real/position/block1', self.pos_block1_callback, 1)
        self.node.create_subscription(Point, '/real/position/block2', self.pos_block2_callback, 1)
        self.node.create_subscription(Point, '/real/position/gripper', self.pos_gripper_callback, 1)
        self.arm_publisher = self.node.create_publisher(JointTrajectory, '/real/joint_trajectory_controller/joint_trajectory', 10)
        self.gripper_publisher = self.node.create_publisher(String, '/real/gripper_control', 10)
    
    def wait_for_move_execution(self):
        time.sleep(0.4)

    def reset(self, seed = None, options=None):
        time.sleep(1.0)
        ## Return observations for RL
        observation = self.get_observation()
        info = {}
        return observation, info

    # def reset(self, seed=None, options=None):
    #     self.current_steps = 0
    #     self.arm_positions = [0.0,0.0,0.0,0.0,0.0,0.0]
    #     self.arm_velocities = [0.0,0.0,0.0,0.0,0.0,0.0]
    #     old_b1 = self.block1_pos
    #     self.gripper_pos = [0.0, 0.0, 0.0] #xyz
    #     self.current_steps = 0

    #     ## Send reset-message to Webots Plugin (ObjectPositionPublisher)
    #     if self.simulation_reset:
    #         # Open Gripper
    #         self.move_gripper("release")
    #         # Move Arm to some random new position
    #         j_1 = np.random.uniform(-0.5, 0.5)
    #         j_2 = np.random.uniform(0.2, 0.5)
    #         j_3 = np.random.uniform(0.1, 0.3)
    #         j_4 = np.random.uniform(0.7, 1.4)
    #         joint_reset_pos = [j_1, j_2, j_3, 0.0, j_4 , 0.0]
    #         self.move_until_position_is_reached(joint_reset_pos)
    #         # Place block at new random position
    #         x_1 = np.random.uniform(0.4, 0.6)
    #         y_1 = np.random.uniform(-0.25, 0.25)
    #         input(f"Place block at {x_1},{y_1}. Press Enter, if done!")
        
    #     ## Return observations for RL
    #     observation = self.get_observation()
    #     info = {}
    #     return observation, info
    
    def move_until_position_is_reached(self, position):
        tolerance = 0.005
        move_command = []
        while not move_command == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]:
            move_command = []
            for i, pos in enumerate(position):
                if pos > self.arm_positions[i] and pos > self.arm_positions[i] + tolerance:
                    move_command.append(0.2)
                elif pos < self.arm_positions[i] and pos < self.arm_positions[i] - tolerance:
                    move_command.append(-0.2)
                else:
                    move_command.append(0.0)
            self.move_arm(move_command)
            print(move_command)
            time.sleep(0.1)
    
    def move_arm(self, velocities):
        scaled_vel = [x * 25 for x in velocities] #Scale up velocity for real robot
        msg = JointTrajectory()
        msg.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        point = JointTrajectoryPoint()
        point.positions = [0.0,0.0,0.0,0.0,0.0,0.0]
        #point.velocities = scaled_vel
        point.velocities = [scaled_vel[0],0.0,scaled_vel[2],scaled_vel[3],scaled_vel[4],scaled_vel[5]]
        msg.points.append(point)
        self.arm_publisher.publish(msg)

    def move_gripper(self, msg):
        send_msg = String()
        send_msg.data = msg
        self.gripper_publisher.publish(send_msg)


def updater(node):
    while True:
        rclpy.spin(node)


def main(args = None):
    ## Initialisitation
    rclpy.init(args=args)
    env1 = RLGoalPosition_real("block1", [0.0, 0.1, 0.1], True)
    env1.distance_reward_active = True


    ## Start spinning nodes
    executor = MultiThreadedExecutor()
    executor.add_node(env1.node)
    Thread(target = executor.spin).start()

    ## Learn position model:
    # modelname = "get_in_goal_pose_v2_"
    # number_of_models = 2
    # RLUtilityClass.learn_position(model_name=modelname, number_of_models=number_of_models, env = env1)

    ##Test grasp success on position learner model:
    # model = DDPG.load("get_in_grasp_pose1_8.zip")
    # test_grasp_from_position_learner(model = model, env = env_grasp_pose)

    # ##Test model:
    model1 = DDPG.load("get_in_goal_pose_0_8.zip")
    model1.set_env(env1)
    # model2 = DDPG.load("get_in_goal_pose_0_8.zip")
    # model2.set_env(env2)
    # model3 = DDPG.load("get_in_goal_pose_0_8.zip")
    # model3.set_env(env3)
    # model4 = DDPG.load("get_in_goal_pose_0_8.zip")
    # model4.set_env(env4)
    vec_env1 = model1.get_env()

    env1.simulation_reset = True
    obs = vec_env1.reset()
    env1.simulation_reset = False
    done = False
    rewards = 0
    while not done and rewards < 100:
        action, _states = model1.predict(obs)
        obs, rewards, done, info = vec_env1.step(action)

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