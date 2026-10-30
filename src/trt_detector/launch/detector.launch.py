import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory("trt_detector"), "config", "detector.yaml")
    args = {
        "engine_path": "engines/yolo11n_fp16_640.plan",
        "image_topic": "/image_raw",
        "publish_annotated_image": "false",
        "video": "",  # set to a video file to also start the test publisher
    }
    decl = [DeclareLaunchArgument(k, default_value=v) for k, v in args.items()]
    detector = Node(
        package="trt_detector", executable="detector_node", output="screen",
        parameters=[cfg, {
            "engine_path": LaunchConfiguration("engine_path"),
            "image_topic": LaunchConfiguration("image_topic"),
            "publish_annotated_image": LaunchConfiguration("publish_annotated_image"),
        }],
    )
    source = Node(
        package="trt_detector", executable="video_publisher", output="screen",
        parameters=[{"video_path": LaunchConfiguration("video"), "image_topic": LaunchConfiguration("image_topic")}],
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration("video"), "' != ''"])),
    )
    return LaunchDescription(decl + [detector, source])
