import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from collections import deque

class JointStateFilter(Node):
    def __init__(self):
        super().__init__('joint_state_filter')
        self.subscription = self.create_subscription(
            JointState,
            '/real/joint_states',
            self.listener_callback,
            10)
        self.publisher = self.create_publisher(JointState, '/real/filtered_joint_states', 10)
        self.__arm_positions = [0.0,0.0,0.0,0.0,0.0,0.0]
        self.__arm_velocities = [0.0,0.0,0.0,0.0,0.0,0.0]

        self.__last_win_size_pos = []
        self.window_size = 10  # Size of the moving average window
        self.data = deque(maxlen=self.window_size)

    def listener_callback(self, joint_state):
        arm_name = joint_state.name
        if len(joint_state.velocity) == 6 and len(joint_state.position) == 6:
            self.__arm_positions = []
            self.__arm_velocities = []
            for i in range(6):
                if i == 0:
                    ind = arm_name.index("joint" + str(i+1))
                    self.__arm_positions.append((joint_state.position[ind] + 0.132)*-1)
                    self.__arm_velocities.append(joint_state.velocity[ind])
                elif True:
                    ind = arm_name.index("joint" + str(i+1))
                    self.__arm_positions.append(joint_state.position[ind])
                    self.__arm_velocities.append(joint_state.velocity[ind])
            
            self.__last_win_size_pos.append(self.__arm_positions)
            if len(self.__last_win_size_pos) > self.window_size:
                self.__last_win_size_pos.pop(0)
                avg_position = [sum(positions)/self.window_size for positions in zip(*self.__last_win_size_pos)]

                filtered_msg = JointState()
                filtered_msg.header = joint_state.header
                filtered_msg.name = ['joint1','joint2','joint3','joint4','joint5','joint6']
                filtered_msg.position = avg_position
                filtered_msg.velocity = joint_state.velocity
                filtered_msg.effort = joint_state.effort
                # Publish the filtered joint states
                self.publisher.publish(filtered_msg)

def main(args=None):
    rclpy.init(args=args)
    joint_state_filter = JointStateFilter()
    rclpy.spin(joint_state_filter)
    joint_state_filter.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
