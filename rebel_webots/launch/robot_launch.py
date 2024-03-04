#  Copyright (c) 2023 Tarik Viehmann
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
import pathlib
import launch
from launch_ros.actions import Node
from launch.substitutions import Command, FindExecutable
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from ament_index_python.packages import get_package_share_directory
from webots_ros2_driver.webots_launcher import WebotsLauncher, Ros2SupervisorLauncher
from webots_ros2_driver.webots_controller import WebotsController
from webots_ros2_driver.utils import controller_url_prefix


def generate_launch_description():
    package_dir = get_package_share_directory('rebel_webots')
    rebel_xacro_file = os.path.join(package_dir, 'urdf', 'rebel.urdf.xacro')
    urdf_output_path = os.path.join(package_dir, 'urdf', 'rebel.urdf')

    # Generate the URDF file from the xacro file
    os.system(f"xacro {rebel_xacro_file} -o {urdf_output_path}")
    robot_description = urdf_output_path
    webots = WebotsLauncher(
        world=os.path.join(package_dir, 'worlds', 'rebel_world.wbt')
    )

    rebel_ros2_control_params=os.path.join(package_dir, 'config', 'rebel_ros2_control_params.yaml')

    rebel_webots_driver = WebotsController(
        robot_name='igus_rebel',
        parameters=[
            {'robot_description': robot_description},
            {"use_sim_time": True},
            {"set_robot_state_publisher": False},
            rebel_ros2_control_params,
        ]
    )
    return LaunchDescription([
        DeclareLaunchArgument(
            'urdf_output_path',
            default_value=os.path.join(get_package_share_directory('rebel_description'), 'urdf', 'rebel.urdf'),
            description='Path to save the generated URDF file'
        ),
        webots,
        rebel_webots_driver,
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=webots,
                on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
            )
        )
    ])
