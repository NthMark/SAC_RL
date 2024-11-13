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
    gazebo_ros_dir = get_package_share_directory('gazebo_ros')

    # Declare launch arguments
    x_pos_arg = DeclareLaunchArgument('x_pos', default_value='0')
    y_pos_arg = DeclareLaunchArgument('y_pos', default_value='0')
    z_pos_arg = DeclareLaunchArgument('z_pos', default_value='0.5')

    # Set robot description parameter
    robot_description_param = os.path.join(myrobot_dir, 'urdf', 'mobile_robot_2.urdf')

    # Define the Gazebo launch description for an empty world with obstacles
    world_path = os.path.join(myrobot_dir, 'worlds', '12obstacles_wall.world')
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_dir, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'world': world_path}.items()
    )

    # Define the node to spawn the URDF model in Gazebo
    spawn_robot_node = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_urdf',
        output='screen',
        arguments=[
            '-param', 'robot_description',
            '-urdf',
            '-model', 'mobile_robot_dqn_2',
            '-x', LaunchConfiguration('x_pos'),
            '-y', LaunchConfiguration('y_pos'),
            '-z', LaunchConfiguration('z_pos')
        ]
    )

    # Define the SAC agent node
    # sac_agent_node = Node(
    #     package='myrobot',
    #     executable='SAC_2act_new.py',
    #     name='mobile_robot_sac',
    #     output='screen'
    # )

    # Assemble the launch description
    return LaunchDescription([
        x_pos_arg,
        y_pos_arg,
        z_pos_arg,
        gazebo_launch,
        spawn_robot_node,
        # sac_agent_node
    ])
