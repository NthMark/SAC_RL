from setuptools import find_packages, setup

package_name = 'rl_sac'
import os
from glob import glob
setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=[package_name, os.path.join(package_name, '*')]),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share',package_name,'launch'), glob('launch/*.launch.py')),
        (os.path.join('share',package_name,'param'), glob('param/*.yaml')),
        (os.path.join('share',package_name,'urdf'), glob('urdf/*.urdf')),
        (os.path.join('share',package_name,'worlds'), glob('worlds/*.world')),
        (os.path.join('share',package_name,'map'), glob('map/*')),
        (os.path.join('share',package_name,'config'), glob('config/*.rviz')),
        (os.path.join('share',package_name,'models','turtlebot3_burger'), glob('models/turtlebot3_burger/*')),
    ],
    package_data={
        package_name: ['analysis/*.csv','common/*']  
    },
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mark',
    maintainer_email='mark@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'training=rl_sac.turtlebot_training:main',
            'env=rl_sac.env_training:main',
            'analyze=rl_sac.analyze:main',
            'pub_test=rl_sac.testing.publisher:main',
            'sub_test=rl_sac.testing.subscriber:main',
            'grid_map=rl_sac.grid_map:main',
            'display_waypoint=rl_sac.display_waypoints:main',
            'path_planning_train=rl_sac.path_planning_train:main',
            'map_data_len=rl_sac.map_data:main',
            'visual_node=rl_sac.common.visual:main'
            # 'bridge=rl_sac.testing.bridge_pub_sub:main'
        ],
    },
)
