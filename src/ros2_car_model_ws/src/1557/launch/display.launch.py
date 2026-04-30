#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    # 获取包路径
    pkg_1557_dir = get_package_share_directory('1557')

    # Launch argument：xacro 文件路径
    model_arg = DeclareLaunchArgument(
        'model',
        default_value=os.path.join(pkg_1557_dir, 'urdf', '1557.xacro'),
        description='Absolute path to robot xacro file'
    )

    # robot_description 参数：使用 xacro 命令生成 URDF
    robot_description = {'robot_description': Command(['xacro ', LaunchConfiguration('model')])}

    return LaunchDescription([
        model_arg,

        # joint_state_publisher_gui 节点
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui'
        ),

        # robot_state_publisher 节点
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[robot_description]
        ),

        # RViz2 节点
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', os.path.join(pkg_1557_dir, 'rviz', 'display.rviz')]
        )
    ])

