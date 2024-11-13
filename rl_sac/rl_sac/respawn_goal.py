import rclpy
from rclpy.node import Node
import random
import time
import os
import math
from gazebo_msgs.srv import SpawnEntity, DeleteEntity,GetModelList
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose
from rclpy.parameter import Parameter
from std_srvs.srv import Empty
class Respawn(Node):
    def __init__(self):
        super().__init__('respawn_node')

        self.model_path = '/home/mark/limo_ws/src/rl_sac/models/goal_box.sdf'
        with open(self.model_path, 'r') as f:
            self.model = f.read()

        self.stage = self.declare_parameter('stage_number', 1).value
        self.goal_position = Pose()

        # if self.stage == 1:
        #     self.init_goal_x, self.init_goal_y = 0.975166, -0.790902
        # elif self.stage == 2:
        #     self.init_goal_x, self.init_goal_y = 2.25, -2.40
        # else:
        self.init_goal_x, self.init_goal_y = -0.18133, 1.3284
        self.last_goal_x = self.init_goal_x
        self.last_goal_y = self.init_goal_y
        self.goal_position.position.x = self.init_goal_x
        self.goal_position.position.y = self.init_goal_y
        # self.goal_position.position.z = 1.0
        self.model_name = 'goal_box'

        self.check_model = False

        self.subscription = self.create_subscription(
            ModelStates,
            'gazebo/model_states',
            self.check_model_callback,
            10
        )

        self.model_list_client=self.create_client(GetModelList,'get_model_list')
        self.spawn_client = self.create_client(SpawnEntity, 'spawn_entity')
        self.delete_client = self.create_client(DeleteEntity, 'delete_entity')

    def check_model_callback(self, model):
        self.check_model = any(name == "goal" for name in model.name)

    def respawn_model(self):
        while not self.model_list_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.future=self.model_list_client.call_async(GetModelList.Request())
        rclpy.spin_until_future_complete(self, self.future)
        self.check_model= any(name == "goal_box" for name in self.future.result().model_names)
        self.get_logger().info('service not available, waiting again...')
        if not self.check_model:
            while not self.spawn_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('service not available, waiting again...')
            self.get_logger().info('spawning ......')
            request = SpawnEntity.Request()
            request.name = self.model_name
            request.xml = self.model
            request.robot_namespace = ''
            request.initial_pose = self.goal_position
            self.spawn_client.call_async(request)
            self.get_logger().info(f"Goal position: {self.goal_position.position.x}, {self.goal_position.position.y}")

    def delete_model(self):
        while not self.model_list_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.future=self.model_list_client.call_async(GetModelList.Request())
        rclpy.spin_until_future_complete(self, self.future)
        self.check_model= any(name == "goal_box" for name in self.future.result().model_names)
        if self.check_model:
            while not self.delete_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('service not available, waiting again...')

            request = DeleteEntity.Request()
            request.name = self.model_name
            self.delete_client.call_async(request)
    def get_start_pose(self):
        self.respawn_model()
        return self.goal_position.position.x, self.goal_position.position.y
    def get_position(self, delete=True):
        if delete:
            self.delete_model()

        goal_x_list = [5.1081,4.0456,2.7147,1.0915,0.37583,-0.18133,-0.14148,-1.4583,-3.6939,-2.8976,-1.5097,-0.73719,-0.35437,3.36,4.295,5.8217,6.5594,6.6926,2.7248]
        goal_y_list = [-9.0511,-9.9685,-9.7536,-9.4915,-4.9754,1.3284,3.7654,3.7568,3.7427,6.2144,6.7212,6.9084,8.079,7.4159,7.4433,7.459,7.686,5.0049,4.9378]
        self.last_goal_x = self.goal_position.position.x
        self.last_goal_y = self.goal_position.position.y
        index=random.randint(0,len(goal_x_list)-1)
        self.goal_position.position.x = goal_x_list[index]
        self.goal_position.position.y = goal_y_list[index]
        while self.goal_position.position.x==self.last_goal_x and self.goal_position.position.y==self.last_goal_y and\
        math.hypot(self.last_goal_x - self.goal_position.position.x, self.goal_position.position.y - self.last_goal_y) >2:
            index=random.randint(0,len(goal_x_list)-1)
            self.goal_position.position.x = goal_x_list[index]
            self.goal_position.position.y = goal_y_list[index]
        time.sleep(0.5)
        self.respawn_model()

        return self.goal_position.position.x, self.goal_position.position.y

def main(args=None):
    rclpy.init(args=args)
    respawn_node = Respawn()
    rclpy.spin(respawn_node)
    respawn_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
