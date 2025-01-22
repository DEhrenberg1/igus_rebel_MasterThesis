How to start up everything that is needed to infer a successful grasp in the real world using this code and the iRC_ROS project:

Go on branch GripperPositionForRealRobot on the latest commit.
Also go on the iRC_ROS project fork that i made and go on the latest commit there as well.
Build and Source these repositories

Start the necessary tools: 

1. With "ros2 launch irc_ros_bringup rebel.launch.py hardware_protocol:=cri use_rviz:=false namespace:=real controller_manager_name:=/real/controller_manager" you can start the connection to the actual Igus ReBeL. If this was successful you should here a 'click' in the robot and you should be able to move the robot via the /real/joint_trajectory_controller/joint_trajectory topic.
   E.g.: "ros2 topic pub /real/joint_trajectory_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory '{header: {stamp: {sec: 0}}, joint_names: ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"], points: [{positions: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], velocities: [0.0, -10.0, 0.0, 0.0, 0.0, 0.0], time_from_start: {sec: 0}}]}' --once"
2. Launch the simulation that always keeps track of the position of the real robot and puts the robot in simulation in the same position. This is used to obtain the gripper pinch position in the real world from the simulation. You can launch this with:
   "ros2 launch rebel_webots helper_for_real_robot_launch.py". Note: This also launches the moving_avg_window.py (creates an average over the last seen joint positions and pushes them on /real/filtered_joint_states) and tf_broadcaster.py which we need for coordination transformation from the camera frame (step 3)
3. Start the position inference using "python3 yolov8_cube_using_tf.py". This starts the camera and pushes coordinates of the rubik's cube onto the respective position topic. (Only works, when tf_broadcaster.py is running).
4. Now you can start the main part: The reinforcement learner. Everything in step 1-3 just provides the needed input and output on ros topics. You can start the reinforcement learner using: "python3 goal_position_learner_real.py"
