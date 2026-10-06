from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package='quantum_mind_ros', executable='human_model_node', name='human_model',
             parameters=[{'refit_every': 20, 'min_answers': 20, 'ask_cost': 1.0, 'error_cost': 5.0}]),
    ])
