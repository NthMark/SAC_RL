import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription,RegisterEventHandler,TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessStart,OnExecutionComplete
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get the package directories
    myrobot_dir = get_package_share_directory('rl_sac')
    map=LaunchConfiguration('map')
    declare_map_cmd=DeclareLaunchArgument(
        'map', default_value=os.path.abspath('/home/mark/limo_ws/src/rl_sac/map/stage_2.yaml'),
        description='map'
    )
    turtlebot_simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(myrobot_dir, 'launch', 'turtlebot3_world.launch.py')
        )
    )
    display_waypoints=Node(
        package='rl_sac',
        executable='display_waypoint',
        name='vel_distribution',
        output='screen'
    )
    grimap_rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(myrobot_dir, 'launch', 'gridmap.launch.py')
        ),
        launch_arguments={'map':map}.items()
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
        declare_map_cmd,
        turtlebot_simulation,
        display_waypoints,
        analysis_node,
        sac_agent_node,
    ])
