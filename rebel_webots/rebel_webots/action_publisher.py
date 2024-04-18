import rclpy
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectoryPoint

def send_goal():
    rclpy.init()
    node = rclpy.create_node('joint_trajectory_client')

    action_client = ActionClient(node, FollowJointTrajectory, '/rebel_arm_controller/follow_joint_trajectory')

    goal_msg = FollowJointTrajectory.Goal()
    goal_msg.trajectory.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
    goal_point = JointTrajectoryPoint()
    goal_point.positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Set desired positions for each joint
    goal_point.velocities = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]  # Set desired velocities for each joint
    goal_point.time_from_start = Duration(sec=0, nanosec=0)  # Set time from start for trajectory point
    goal_msg.trajectory.points.append(goal_point)

    future = action_client.send_goal_async(goal_msg)
    rclpy.spin_until_future_complete(node, future)
    if future.result() is not None:
        response = future.result()
        print(response)
        if response.accepted:
            node.get_logger().info('Goal accepted!')
        else:
            node.get_logger().warning('Goal rejected!')
    else:
        node.get_logger().error('Goal request failed!')

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    send_goal()
