#!/usr/bin/env python3
"""Launch the object-finder service and the pick / place action servers."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ── arguments ──────────────────────────────────────────────────────────
    camera_topic      = LaunchConfiguration('camera_topic')
    depth_topic       = LaunchConfiguration('depth_topic')
    camera_info_topic = LaunchConfiguration('camera_info_topic')
    camera_frame      = LaunchConfiguration('camera_frame')
    reference_frame   = LaunchConfiguration('reference_frame')
    device            = LaunchConfiguration('device')

    # ── nodes ───────────────────────────────────────────────────────────────
    object_finder_server = Node(
        package='manipulation_servers',
        executable='object_finder_server',
        name='object_finder_server',
        output='screen',
        parameters=[{
            'camera_topic':      camera_topic,
            'depth_topic':       depth_topic,
            'camera_info_topic': camera_info_topic,
            'camera_frame':      camera_frame,
            'reference_frame':   reference_frame,
            'device':            device,
        }],
    )

    pick_server = Node(
        package='manipulation_servers',
        executable='pick_server',
        name='pick_object_server',
        output='screen',
        parameters=[{
            'reference_frame':          reference_frame,
            'default_pre_grasp_height': 0.1,
        }],
    )

    place_server = Node(
        package='manipulation_servers',
        executable='place_server',
        name='place_object_server',
        output='screen',
        parameters=[{
            'reference_frame':    reference_frame,
            'default_drop_height': 0.15,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('camera_topic',      default_value='/camera/camera/color/image_raw'),
        DeclareLaunchArgument('depth_topic',       default_value='/camera/camera/aligned_depth_to_color/image_raw'),
        DeclareLaunchArgument('camera_info_topic', default_value='/camera/camera/color/camera_info'),
        DeclareLaunchArgument('camera_frame',      default_value='camera_color_optical_frame'),
        DeclareLaunchArgument('reference_frame',   default_value='link_base'),
        DeclareLaunchArgument('device',            default_value='cuda'),
        object_finder_server,
        pick_server,
        place_server,
    ])
