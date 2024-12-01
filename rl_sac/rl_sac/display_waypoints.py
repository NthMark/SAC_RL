import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point,PoseStamped, PoseWithCovarianceStamped
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import OccupancyGrid
import time
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
import rclpy.wait_for_message
from tf2_ros import TransformBroadcaster,StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math
import numpy as np
from .grid_map import GridMap
from std_msgs.msg import Float32MultiArray
import matplotlib.pyplot as plt
from .common.config import ACTION_V_MAX,ACTION_V_MIN,ACTION_W_MAX,ACTION_W_MIN
class WaypointPublisher(Node):
    def __init__(self):
        super().__init__('waypoint_publisher')
        self.publisher = self.create_publisher(MarkerArray, 'static_obstacle', 10)
        self.pub_vel_linear = self.create_publisher(Marker, 'pub_vel_linear', 10)
        self.pub_vel_angular = self.create_publisher(Marker, 'pub_vel_angular', 10)
        self.timer = self.create_timer(1.0, self.publish_waypoints)
        self.counter = 0  # For dynamic visualization
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_broadcaster = StaticTransformBroadcaster(self)
        self.pub_goal = self.create_publisher(PoseStamped, 'goal_pose', 10)
        self.pub_initial = self.create_publisher(PoseWithCovarianceStamped, 'initialpose', 10)
        self.sub_vel_distribute=self.create_subscription(
            Float32MultiArray,
            'vel_distribute',
            self.vel_distribution_callback,
            10
        )
        gridmap=GridMap(debug=True,is_publish=False)
        # idx = int(math.floor((0 - self.map['origin'].position.y) / self.map['resolution']) * self.map['width'] + math.floor((0 - self.map['origin'].position.x) / self.map['resolution']))
        # self.get_logger().info(f"{self.map['data'][idx]}")
        # Example waypoints
        self.waypoints_first_quadrans  = []
        self.waypoints_second_quadrans = []
        self.waypoints_third_quadrans  = []
        self.waypoints_fourth_quadrans = []
        self.vel_linear=[]
        self.vel_angular=[]
        self.pos=[]
        self.colors_linear=[]
        self.colors_angular=[]
        width_offset=10
        heigh_offset=0
        self.resolution=gridmap.resolution
        for i_x in np.linspace(0,(gridmap.width+width_offset)//2*gridmap.resolution,(gridmap.width+width_offset)//2+1):
            for i_y in np.linspace(0,gridmap.height//2*gridmap.resolution,gridmap.height//2+1):
                temp_occ=gridmap.occ_grid_value(i_x,i_y)
                if temp_occ ==100 or temp_occ==-1:
                    p = Point()
                    p.x, p.y = i_x,i_y
                    p.z = 0.0
                    self.waypoints_first_quadrans.append(p)
        for i_x in np.linspace(0,(gridmap.width+width_offset)//2*gridmap.resolution,(gridmap.width+width_offset)//2+1):
            for i_y in np.linspace(0,-(gridmap.height//2)*gridmap.resolution,gridmap.height//2+1):
                temp_occ=gridmap.occ_grid_value(i_x,i_y)
                if temp_occ== -1 or temp_occ ==100:
                    p = Point()
                    p.x, p.y = i_x,i_y
                    p.z = 0.0
                    self.waypoints_second_quadrans.append(p)
        for i_x in np.linspace(0,-(gridmap.width//2-width_offset)*gridmap.resolution,(gridmap.width-width_offset)//2+1):
            for i_y in np.linspace(0,gridmap.height//2*gridmap.resolution,gridmap.height//2+1):
                temp_occ=gridmap.occ_grid_value(i_x,i_y)
                if temp_occ== -1 or temp_occ ==100:
                    p = Point()
                    p.x, p.y = i_x,i_y
                    p.z = 0.0
                    self.waypoints_third_quadrans.append(p)
        for i_x in np.linspace(0,-(gridmap.width//2-width_offset)*gridmap.resolution,(gridmap.width-width_offset)//2+1):
            for i_y in np.linspace(0,-(gridmap.height//2)*gridmap.resolution,gridmap.height//2+1):
                temp_occ=gridmap.occ_grid_value(i_x,i_y)
                if temp_occ== -1 or temp_occ ==100:
                    p = Point()
                    p.x, p.y = i_x,i_y
                    p.z = 0.0
                    self.waypoints_fourth_quadrans.append(p)    
        # self.waypoints = [
        #     (0.0, 0.0),
        #     (1.0, 1.0),
        #     (2.0, 2.0),
        #     (3.0, 1.5),
        #     (4.7, 3.0),
        #     (-6.59,-11.3)
        # ]
        self.marker_first = Marker()
        self.marker_first.header.frame_id = "map"
        self.marker_first.header.stamp = self.get_clock().now().to_msg()
        self.marker_first.ns = "first_occ_quad"
        self.marker_first.id = 0
        self.marker_first.type = Marker.POINTS
        self.marker_first.action = Marker.ADD

        self.marker_second = Marker()
        self.marker_second.header.frame_id = "map"
        self.marker_second.header.stamp = self.get_clock().now().to_msg()
        self.marker_second.ns = "second_occ_quad"
        self.marker_second.id = 1
        self.marker_second.type = Marker.POINTS
        self.marker_second.action = Marker.ADD

        self.marker_third = Marker()
        self.marker_third.header.frame_id = "map"
        self.marker_third.header.stamp = self.get_clock().now().to_msg()
        self.marker_third.ns = "third_occ_quad"
        self.marker_third.id = 2
        self.marker_third.type = Marker.POINTS
        self.marker_third.action = Marker.ADD

        self.marker_fourth = Marker()
        self.marker_fourth.header.frame_id = "map"
        self.marker_fourth.header.stamp = self.get_clock().now().to_msg()
        self.marker_fourth.ns = "fourth_occ_quad"
        self.marker_fourth.id = 3
        self.marker_fourth.type = Marker.POINTS
        self.marker_fourth.action = Marker.ADD

        self.marker_robot=Marker()
        self.marker_robot.header.frame_id = "map"
        self.marker_robot.header.stamp = self.get_clock().now().to_msg()
        self.marker_robot.ns = "robot"
        self.marker_robot.id = 4
        self.marker_robot.type = Marker.SPHERE
        self.marker_robot.action = Marker.ADD
        self.marker_robot.pose.position.x=0.0
        self.marker_robot.pose.position.y=0.0
        self.marker_robot.pose.position.z=0.0
        self.marker_robot.scale.x = 0.14  # Scale of the sphere in meters
        self.marker_robot.scale.y = 0.14
        self.marker_robot.scale.z = 0.14
        self.marker_robot.color.r = 0.0
        self.marker_robot.color.g = 1.0
        self.marker_robot.color.b = 0.0
        self.marker_robot.color.a = 1.0  # Fully opaque
        # Set self.marker_first properties
        color=Marker().color
        scale=Marker().scale
        scale.x = gridmap.resolution  # Size of the points
        scale.y = gridmap.resolution
        color.a = 1.0  # Alpha (transparency)
        color.r = 1.0
        color.g = 0.0
        color.b = 0.0
        self.marker_first.scale =scale
        self.marker_first.color =color
        self.marker_second.scale =scale
        self.marker_second.color =color
        self.marker_third.scale =scale
        self.marker_third.color =color
        self.marker_fourth.scale =scale
        self.marker_fourth.color =color
        self.marker_first.points=self.waypoints_first_quadrans
        self.marker_second.points=self.waypoints_second_quadrans
        self.marker_third.points=self.waypoints_third_quadrans
        self.marker_fourth.points=self.waypoints_fourth_quadrans

        self.marker_test = Marker()
        self.marker_test.header.frame_id = "map"
        self.marker_test.header.stamp = self.get_clock().now().to_msg()
        self.marker_test.ns = "test"
        self.marker_test.id = 4
        self.marker_test.type = Marker.POINTS
        self.marker_test.action = Marker.ADD
        self.marker_test.scale.x = 0.14  # Scale of the sphere in meters
        self.marker_test.scale.y = 0.14
        self.marker_test.scale.z = 0.14
        self.marker_test.color.r = 1.0
        self.marker_test.color.g = 1.0
        self.marker_test.color.b = 0.0
        self.marker_test.color.a = 1.0 

        self.marker_vel_linear = Marker()
        self.marker_vel_linear.header.frame_id = "map"
        self.marker_vel_linear.header.stamp = self.get_clock().now().to_msg()
        self.marker_vel_linear.ns = "vel_linear"
        self.marker_vel_linear.id = 5
        self.marker_vel_linear.type = Marker.POINTS
        self.marker_vel_linear.action = Marker.ADD
        self.marker_vel_linear.scale = scale 

        self.marker_vel_angular = Marker()
        self.marker_vel_angular.header.frame_id = "map"
        self.marker_vel_angular.header.stamp = self.get_clock().now().to_msg()
        self.marker_vel_angular.ns = "vel_angular"
        self.marker_vel_angular.id = 6
        self.marker_vel_angular.type = Marker.POINTS
        self.marker_vel_angular.action = Marker.ADD
        self.marker_vel_angular.scale=scale
        self.marker_occ_arr =MarkerArray()
        # p=Point()
        # p.x=0.0#gridmap.width//2*gridmap.resolution
        # p.y=gridmap.height//2*gridmap.resolution
        # p.z=0.0
        # self.random_start=[(0,0),(0,4),(2,2),(-2,-2),(3,-5),(0,7),(7,2),(7,3),(-2,7)]
        # haizz=[]
        # for ele in self.random_start:
        #     p=Point()
        #     p.x=float(ele[0])
        #     p.y=float(ele[1])
        #     p.z=0.0
        #     haizz.append(p)
        # self.marker_test.points=haizz
        self.marker_occ_arr.markers=[self.marker_first,self.marker_second,self.marker_third,self.marker_fourth,self.marker_robot]
    def round_to_nearest_half(self,value,resolution):
        return round(value / resolution) * resolution
    def map_altitude_to_color(self,altitude,altitude_max,altitude_min):
        # Map altitude value from range (-2, 2) to (0, 1)
        a, b = altitude_min, altitude_max  # Altitude range
        c, d = 0, 1   # Color value range
        
        # Mapping formula
        color_value = c + ((altitude - a) / (b - a)) * (d - c)
        
        # Ensure color_value is within bounds (0, 1)
        color_value = min(max(color_value, 0), 1)
        
        return color_value
    def altitude_to_rgb(self,altitude,altitude_max,altitude_min):
    # Normalize altitude to range (0, 1)
        normalized_value = self.map_altitude_to_color(altitude,altitude_max,altitude_min)

        # Use a color map to generate RGB values (e.g., 'coolwarm' color map from matplotlib)
        cmap = plt.get_cmap('coolwarm')  # You can use different color maps such as 'viridis', 'plasma', etc.
        r, g, b, _ = cmap(normalized_value)  # Get RGBA, we only need RGB
        
        return r, g, b

    def vel_distribution_callback(self,msg):
        pos=Point()
        color_linear=Marker().color
        color_linear.a=1.0
        color_angular=Marker().color
        color_angular.a=1.0
        pos.x=self.round_to_nearest_half(msg.data[-2],self.resolution)
        pos.y=self.round_to_nearest_half(msg.data[-1],self.resolution)
        vel_linear=msg.data[0]
        vel_angular=msg.data[1]
        color_linear.r,color_linear.g,color_linear.b=self.altitude_to_rgb(vel_linear,ACTION_V_MAX,ACTION_V_MIN)
        color_angular.r,color_angular.g,color_angular.b=self.altitude_to_rgb(vel_angular,ACTION_W_MAX,ACTION_W_MIN)
        if pos not in self.pos:
            self.pos.append(pos)
            self.colors_angular.append(color_angular)
            self.colors_linear.append(color_linear)
        else:
            idx=self.pos.index(pos)
            self.colors_angular[idx]=color_angular
            self.colors_linear[idx]=color_linear

        self.marker_vel_linear.points=self.pos
        self.marker_vel_linear.colors=self.colors_linear
        self.marker_vel_angular.points=self.pos
        self.marker_vel_angular.colors=self.colors_angular

        self.pub_vel_linear.publish(self.marker_vel_linear)
        self.pub_vel_angular.publish(self.marker_vel_angular)
    def publish_waypoints(self):
        self.publisher.publish(self.marker_occ_arr)
        # # Add waypoints dynamically (simulate training progress)
        # points_free = []
        # points_occ= []
        # colors = []
        # for i in range(min(self.counter, len(self.waypoints))):
        #     p = Point()
        #     p.x, p.y = self.waypoints[i][0],self.waypoints[i][1]
        #     p.z = 0.0
        #     if self.waypoints[i][2]==-1 or self.waypoints[i][2]==100:
        #         points_occ.append(p)
        #     elif self.waypoints[i][2]==0:
        #         points_free.append(p)
        # self.marker_first.points = points_free
        # self.marker_second.points  = points_occ
        # Simulate adding waypoints over time
         # p = Point()
        # p.x, p.y = self.waypoints[self.counter][0],self.waypoints[self.counter][1]
        # p.z = 0.0
        # points.append(p)
        # if self.marker_first.points
        # Publish the self.marker_first
        # if self.counter < len(self.waypoints):
        #     self.counter += 1
    def set_initial_pose(self, x,y,z,roll,pitch,yaw):
        rpy_pose = PoseWithCovarianceStamped()
        rpy_pose.header.frame_id = 'map'
        rpy_pose.header.stamp = self.get_clock().now().to_msg()
        rpy_pose.pose.pose.position.x = x
        rpy_pose.pose.pose.position.y = y
        rpy_pose.pose.pose.position.z = z
        qx, qy, qz,qw = self.quaternion_from_euler(roll, pitch, yaw)
        
        rpy_pose.pose.pose.orientation.x = qx
        rpy_pose.pose.pose.orientation.y = qy
        rpy_pose.pose.pose.orientation.z = qz
        rpy_pose.pose.pose.orientation.w = qw
        
        self.pub_initial.publish(rpy_pose)
        self.get_logger().info(f'Published initial_2d: x={rpy_pose.pose.pose.position.x}, y={rpy_pose.pose.pose.position.y}, yaw={rpy_pose.pose.pose.position.z},\
                               qx={rpy_pose.pose.pose.orientation.x},qy={rpy_pose.pose.pose.orientation.y},qz={rpy_pose.pose.pose.orientation.z},qw={rpy_pose.pose.pose.orientation.w}')

    def quaternion_from_euler(self,ai, aj, ak):
        ai /= 2.0
        aj /= 2.0
        ak /= 2.0
        ci = math.cos(ai)
        si = math.sin(ai)
        cj = math.cos(aj)
        sj = math.sin(aj)
        ck = math.cos(ak)
        sk = math.sin(ak)
        cc = ci*ck
        cs = ci*sk
        sc = si*ck
        ss = si*sk

        q = np.empty((4, ))
        q[0] = cj*sc - sj*cs
        q[1] = cj*ss + sj*cc
        q[2] = cj*cs - sj*sc
        q[3] = cj*cc + sj*ss

        return q[0],q[1],q[2],q[3]
def main(args=None):
    rclpy.init(args=args)
    waypoint_publisher = WaypointPublisher()
    waypoint_publisher.set_initial_pose(-1.899293676243032, -0.5, 0.04251584008924808, \
                                        0.0,0.0,0.0)
    try:
        rclpy.spin(waypoint_publisher)
    except KeyboardInterrupt:
        pass

    waypoint_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
