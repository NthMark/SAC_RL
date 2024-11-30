import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get the package directories
    myrobot_dir = get_package_share_directory('rl_sac')
    turtlebot_simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(myrobot_dir, 'launch', 'turtlebot3_world.launch.py')
        )
    )
    grimap_rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(myrobot_dir, 'launch', 'gridmap.launch.py')
        ),
        launch_arguments={'map':'src/rl_sac/map/stage_2.yaml'}.items()
    )
    gridmap_node=Node(
        package='rl_sac',
        executable='grid_map',
        name='gridmap',
        output='screen'
    )
    env_node=Node(
        package='rl_sac',
        executable='env',
        name='environment',
        output='screen'
    )
    # Define the SAC agent node
    analysis_node=Node(
        package='rl_sac',
        executable='analyze',
        name='analysis',
        output='screen'
    )
    sac_agent_node = Node(
        package='rl_sac',
        executable='training',
        name='sac_navigation',
        output='screen'
    )

    # Assemble the launch description
    return LaunchDescription([
        turtlebot_simulation,
        grimap_rviz,
        gridmap_node,
        env_node,
        analysis_node,
        sac_agent_node,
    ])
