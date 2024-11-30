import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose
import random
import math
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import copy
class GridMap(Node):
    def __init__(self, neighbor_size=5, debug=False,is_publish=True):
        super().__init__('grid_map')
        self.neighbor_size = neighbor_size
        self.debug = debug
        self.occ_map_data = None
        self.resolution = None
        self.origin_x = None
        self.origin_y = None
        self.width = None
        self.height = None
        self.goal_pose=None
        self.id=0
        self.time_step=0
        self.NotSetUpInitial=True
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )
        self.random_start=[(0,0),(0,4),(2,2),(-2,-2),(3,-5),(0,7),(7,2),(7,3),(-2,7),(3,7)]
        self.random_goal =[(0,0),(0,4),(2,2),(-2,-2),(3,-5),(0,7),(7,2),(7,3),(-2,7),(3,7)]
        # Subscriber to Occupancy Grid map
        self.occ_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.occ_map_callback,
            qos_profile
        )
        # Wait until map data is received
        while rclpy.ok() and self.occ_map_data is None:
            rclpy.spin_once(self)
        if self.debug:
            self.get_occ_map_info()
        # self.generate_start_goal()
        if is_publish:
            self.get_logger().info(f"Publishing Marker enabled")
            self.pub_waypoints = self.create_publisher(MarkerArray, 'current_waypoints', 10)
            self.timer = self.create_timer(0.5, self.publish_waypoints)
            self.marker_waypoints = Marker()
            self.marker_waypoints.header.frame_id = "map"
            self.marker_waypoints.header.stamp = self.get_clock().now().to_msg()
            self.marker_waypoints.ns = "current_waypoints"
            self.marker_waypoints.id = 1
            self.marker_waypoints.type = Marker.POINTS
            self.marker_waypoints.action = Marker.ADD
            self.marker_waypoints.scale.x=self.resolution
            self.marker_waypoints.scale.y=self.resolution
            self.marker_waypoints.color.a = 1.0  # Alpha (transparency)
            self.marker_waypoints.color.r = 0.0
            self.marker_waypoints.color.g = 1.0
            self.marker_waypoints.color.b = 0.0
            self.waypoints= []

            self.marker_start = Marker()
            self.marker_start.header.frame_id = "map"
            self.marker_start.header.stamp = self.get_clock().now().to_msg()
            self.marker_start.ns = "start"
            self.marker_start.id = 2
            self.marker_start.type = Marker.POINTS
            self.marker_start.action = Marker.ADD
            self.marker_start.scale.x=self.resolution
            self.marker_start.scale.y=self.resolution
            self.marker_start.color.a = 1.0  # Alpha (transparency)
            self.marker_start.color.r = 1.0
            self.marker_start.color.g = 1.0
            self.marker_start.color.b = 0.0
            self.start_points= []

            self.marker_goal = Marker()
            self.marker_goal.header.frame_id = "map"
            self.marker_goal.header.stamp = self.get_clock().now().to_msg()
            self.marker_goal.ns = "goal"
            self.marker_goal.id = 3
            self.marker_goal.type = Marker.POINTS
            self.marker_goal.action = Marker.ADD
            self.marker_goal.scale.x=self.resolution
            self.marker_goal.scale.y=self.resolution
            self.marker_goal.color.a = 1.0  # Alpha (transparency)
            self.marker_goal.color.r = 0.0
            self.marker_goal.color.g = 0.0
            self.marker_goal.color.b = 1.0
            self.goal_points= []
            self.marker_occ_arr =MarkerArray()
    def occ_map_callback(self, msg):
        self.occ_map_data = msg
        self.resolution = msg.info.resolution
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y
        self.width = msg.info.width
        self.height = msg.info.height
        self.robot_radius=self.neighbor_size*self.resolution
    def publish_waypoints(self):
        # self.get_logger().info(f"marker_occ_arr: {self.marker_occ_arr}")
        self.pub_waypoints.publish(self.marker_occ_arr)
    def get_occ_map_info(self):
        self.get_logger().info('Get Occupancy Map Information:')
        self.get_logger().info(f' >> frame_id: {self.occ_map_data.header.frame_id}')
        self.get_logger().info(f' >> resolution: {self.occ_map_data.info.resolution:.4f}')
        self.get_logger().info(f' >> size: ({self.occ_map_data.info.height}, {self.occ_map_data.info.width})')
        self.get_logger().info(f' >> origin: ({self.occ_map_data.info.origin.position.x:.2f}, {self.occ_map_data.info.origin.position.y:.2f})')

    def get_map_info_value(self):
        return self.origin_x, self.origin_y, self.resolution, self.width, self.height

    def get_map_frame(self):
        return self.occ_map_data.header.frame_id

    def occ_grid_value(self, x, y):
        idx = int(math.floor((y - self.origin_y) / self.resolution) * self.width + math.floor((x - self.origin_x) / self.resolution))
        if idx >= len(self.occ_map_data.data):  # out of range --> unknown
            return -1
        return self.occ_map_data.data[idx]
    def get_local_occupancy_window(self, window_size):
        # Get a local occupancy window around the current position
        half_window = window_size // 2
        x, y = self.current_pose.position.x,self.current_pose.position.y
        local_window = []

        for dx in range(-half_window, half_window + 1):
            for dy in range(-half_window, half_window + 1):
                local_window.append(self.occ_grid_value(x+dx*self.resolution,y+dy*self.resolution))

        return np.array(local_window)

    def is_obstacle_point(self, x, y):
        polygon_cells = self.rectangle_fill_cells(x + self.robot_radius, y + self.robot_radius, x - self.robot_radius, y - self.robot_radius)
        if not polygon_cells:
            self.get_logger().info('Out of Range.')
            return True
        for point in polygon_cells:
            idx = int(math.floor((point[1] - self.origin_y) / self.resolution) * self.width + math.floor((point[0] - self.origin_x) / self.resolution))
            if idx >= len(self.occ_map_data.data):
                occupancy = -1
            else:
                occupancy = self.occ_map_data.data[idx]
            if occupancy == 100 or occupancy == -1:
                if self.debug:
                    self.get_logger().info(f'Occupied at ({point[0]}, {point[1]})')
                return True
        return False

    def rectangle_fill_cells(self, max_x, max_y, min_x, min_y):
        if min_x < self.origin_x or min_y < self.origin_y or max_x > self.origin_x + self.width * self.resolution or max_y > self.origin_y + self.height * self.resolution:
            return []

        polygon_cells = []
        x = min_x
        while x < max_x:
            y = min_y
            while y < max_y:
                polygon_cells.append((x, y))
                y += self.resolution
            x += self.resolution

        return polygon_cells

    def get_random_value(self, min_val, max_val):
        return random.uniform(min_val, max_val)

    # def is_obstacle_between(self, ax, ay, bx, by):
    #     dist_ab = math.hypot(bx - ax, by - ay)
    #     if dist_ab >= self.robot_radius:
    #         steps_number = int(math.floor(dist_ab / self.robot_radius))
    #         theta = math.atan2(by - ay, bx - ax)

    #         for n in range(steps_number):
    #             wx = ax + n * self.robot_radius * math.cos(theta)
    #             wy = ay + n * self.robot_radius * math.sin(theta)
    #             if self.is_obstacle_point(wx, wy):
    #                 if self.debug:
    #                     self.get_logger().info('There is an obstacle between the two nodes.')
    #                 return True
    #     return self.is_obstacle_point(bx, by)

    # def is_obstacle_around(self, x, y, step=1):
    #     for dy in range(-step, step + 1):
    #         for dx in range(-step, step + 1):
    #             if self.is_obstacle_point(x + dx * self.resolution, y + dy * self.resolution):
    #                 if self.debug:
    #                     self.get_logger().info('There is an obstacle around the node.')
    #                 return True
    #     return False

    def step(self,action):
        self.time_step+=1
        next_position=copy.deepcopy(self.current_pose.position)
        done = False
        reach_goal=False
        if action == 0:  # Move up
            next_position.x += self.resolution
        elif action == 1:  # Move down
            next_position.x -= self.resolution
        elif action == 2:  # Move left
            next_position.y += self.resolution
        elif action == 3:  # Move right
            next_position.y -= self.resolution
        if self.occ_grid_value(next_position.x,next_position.y) == -1 or self.occ_grid_value(next_position.x,next_position.y) ==100:
            done =True
        if self.is_reach_goal(next_position):
            reach_goal=True
        next_position.z=0.0
        self.current_pose.position=copy.deepcopy(next_position)
        state=self.get_state(self.current_pose)
        reward=self.get_reward(done,reach_goal)
        # if reward == REWARD:
        #     for x_i,y_i in self.waypoints:
        #         if x_i == next_position.x and y_i == next_position.y:
        #             reward=-15
        #         break
        self.waypoints.append(next_position)
        # self.get_logger().info(f"waypoint: {self.waypoints}")
        # if len(self.waypoints)>3:
        #     self.get_logger().info("YEal kjsdfskadfjhasfdlasud")
        #     self.marker_waypoints.points=self.waypoints[:3]
        # else:
        #     self.marker_waypoints.points=self.waypoints[:1]
        self.marker_waypoints.points=self.waypoints
        self.marker_occ_arr.markers=[self.marker_waypoints,self.marker_start,self.marker_goal]
        self.pub_waypoints.publish(self.marker_occ_arr)
        return (state,reward,done)
    def get_reward(self,done,reach_goal):
        reward=-2
        if done:
            reward= -30
        if reach_goal:
            self.get_logger().info('Goal!!')
            self.generate_start_goal()
            reward = 200
        return reward
    def calculate_reward(self,point,goal):
        reward=-2
        window_size=self.neighbor_size
        half_window = window_size // 2
        distance=round(math.hypot(point.y-goal.y , point.x-goal.x), 2)

        if distance<=half_window*self.resolution:
            reward = 200
        return reward
    def is_reach_goal(self, position):
        window_size=self.neighbor_size
        half_window = window_size // 2
        distance=round(math.hypot(self.goal.position.y-position.y , self.goal.position.x-position.x), 2)

        if distance<=half_window*self.resolution:
            return True
        return False
                


    def get_state(self,pose):
        # local_neighbor=self.get_local_occupancy_window(self.neighbor_size)
        distance=round(math.hypot(self.goal.position.y-pose.position.y , self.goal.position.x-pose.position.x), 2)
        return np.concatenate([np.array([pose.position.x,pose.position.y]),np.array([distance])])
    def reset(self):
        self.waypoints=[]
        self.current_pose=copy.deepcopy(self.start)
        self.waypoints.append(self.current_pose.position)
        self.marker_waypoints.points=self.waypoints
        self.marker_occ_arr.markers=[self.marker_waypoints,self.marker_start,self.marker_goal]
        self.pub_waypoints.publish(self.marker_occ_arr)
        state=self.get_state(self.start)
        return np.asarray(state)

    def generate_start_goal(self): #Cannot initialize function in __init__
        self.start = Pose()
        self.goal = Pose()
        # Generate random start point
        # while True:
            # start_x = self.get_random_value(self.origin_x, self.origin_x + self.width * self.resolution)
            # start_y = self.get_random_value(self.origin_y, self.origin_y + self.height * self.resolution)
            # if not self.is_obstacle_point(start_x, start_y):
            #     break
        idx=random.randint(0,len(self.random_start)-1)
        start_x=self.random_start[idx][0]
        start_y=self.random_start[idx][1]
        self.start.position.x = float(start_x)
        self.start.position.y = float(start_y)
        self.start.position.z = 0.0
        self.start_points.append(self.start.position)
        self.marker_start.points=self.start_points
        # self.start.header.frame_id = self.get_map_frame()
        # self.start.header.stamp = self.get_clock().now().to_msg()
        # self.start.header.seq = self.id
        self.current_pose=copy.deepcopy(self.start)
        # Generate random goal point
        # while True:
        #     goal_x = self.get_random_value(self.origin_x, self.origin_x + self.height * self.resolution)
        #     goal_y = self.get_random_value(self.origin_y, self.origin_y + self.width * self.resolution)
        #     if not self.is_obstacle_point(goal_x, goal_y) and (goal_x != start_x or goal_y != start_y):
        #         break
        while True:
            idx=random.randint(0,len(self.random_goal)-1)
            goal_x = self.random_goal[idx][0]
            goal_y = self.random_goal[idx][1]
            if goal_x != start_x or goal_y != start_y:
                break
        self.goal.position.x = float(goal_x)
        self.goal.position.y = float(goal_y)
        self.goal.position.z = 0.0
        self.goal_points.append(self.goal.position)
        self.marker_goal.points=self.goal_points
        self.marker_occ_arr.markers=[self.marker_waypoints,self.marker_start,self.marker_goal]
        self.pub_waypoints.publish(self.marker_occ_arr)
        # self.goal.header.frame_id = self.get_map_frame()
        # self.goal.header.stamp = self.get_clock().now().to_msg()
        # self.goal.header.seq = self.id
        self.id+=1
        if self.debug:
            self.get_logger().info(f'Generated Start Point: ({start_x}, {start_y})')
            self.get_logger().info(f'Generated Goal Point: ({goal_x}, {goal_y})')

def main(args=None):
    rclpy.init(args=args)
    grid_map = GridMap(debug=True,is_publish=True)
    rclpy.spin(grid_map)
    grid_map.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
