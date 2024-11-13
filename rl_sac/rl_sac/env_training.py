import rclpy
from rclpy.node import Node
import numpy as np
import math
import time
import rclpy.wait_for_message
import tf2_ros
from transforms3d.euler import quat2euler
from math import pi, sqrt, pow
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from .respawn_goal import Respawn
class Env(Node):
    def __init__(self):
        super().__init__('env')
        self.delete = False
        self.respawn_goal = Respawn()
        self.goal_x, self.goal_y = self.respawn_goal.get_start_pose()
        self.heading = 0
        self.get_goalbox = False
        self.position = Point()
        self.line_error = 0
        self.current_obstacle_angle = 0
        self.old_obstacle_angle = 0
        self.current_obstacle_min_range = 0
        self.old_obstacle_min_range = 0
        self.t = 0
        self.old_t = time.time()
        self.dt = 0
        self.time_step = 0

        self.reset_proxy = self.create_client(Empty, 'reset_simulation')
        self.unpause_proxy = self.create_client(Empty, 'unpause_physics')
        self.pause_proxy = self.create_client(Empty, 'pause_physics')

        self.pub_cmd_vel = self.create_publisher(Twist, 'cmd_vel', 10)
        # self.sub_odom = self.create_subscription(Odometry, 'odom', self.get_odometry, 10) 
        # self.sub_scan = self.create_subscription(LaserScan, 'scan', self.get_scan_data, 10)
        self.scan = LaserScan()
    # def get_scan_data(self, scan_msg:LaserScan):-> didnt work
    #     # self.get_logger.info(scan_msg)
    #     self.scan = scan_msg
    #     if len(self.scan.ranges) > 0:
    #         self.get_logger().info(f"--------------------{self.scan}")

    def get_goal_distance(self,position):
        return round(math.hypot(self.goal_x - position.x, self.goal_y - position.y), 4)

    # def get_odometry(self, odom):-> didnt work
    #     self.position = odom.pose.pose.position
    #     orientation = odom.pose.pose.orientation
    #     orientation_list = [orientation.w,orientation.x, orientation.y, orientation.z]
    #     _, _, yaw = quat2euler(orientation_list)

    #     goal_angle = math.atan2(self.goal_y - self.position.position.y, self.goal_x - self.position.position.x)
    #     self.heading = goal_angle - yaw
    #     if self.heading > pi:
    #         self.heading -= 2 * pi
    #     elif self.heading < -pi:
    #         self.heading += 2 * pi
    def step(self, action, ep):
        
        self.time_step += 1
        
        vel_cmd = Twist()
        vel_cmd.linear.x  =  float(action[1]) #0.5
        vel_cmd.angular.z = float(action[0]) # action
        self.pub_cmd_vel.publish(vel_cmd)
        
        #print("EP:", ep, " Step:", t, " Goal_x:",self.goal_x, "  Goal_y:",self.goal_y)
        data=None
        while data is None:
            try:
                data = rclpy.wait_for_message.wait_for_message(LaserScan,self,'scan', qos_profile=10) #(boolean,message)
            except Exception as e:
                self.get_logger().warn(f"Failed to receive LaserScan message: {e}")
        state, done, arrival = self.get_state(data[1], ep)
        reward = self.setReward(state, done, action)
        
        return np.asarray(state), reward, done, arrival 
    def get_state(self, scan, ep):
        ###########
        odom=None
        while odom is None:
            try:
                odom = rclpy.wait_for_message.wait_for_message(Odometry,self,'odom', qos_profile=10) #(boolean,message)
            except Exception as e:
                self.get_logger().warn(f"Failed to receive LaserScan message: {e}")
        self.position = odom[1].pose.pose.position
        orientation = odom[1].pose.pose.orientation
        orientation_list = [orientation.w,orientation.x, orientation.y, orientation.z]
        _, _, yaw = quat2euler(orientation_list)

        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)
        self.heading = goal_angle - yaw
        if self.heading > pi:
            self.heading -= 2 * pi
        elif self.heading < -pi:
            self.heading += 2 * pi
        #########
        scan_range=[12 if r==float('Inf') else 0 if np.isnan(r) else r for r in scan.ranges]
        obstacle_min_range = round(min(scan_range), 2)
        obstacle_angle = np.argmin(scan_range)
        self.get_logger().info(f'Current pose: \n x:{self.position.x} y:{self.position.y}')
        self.get_logger().info(f'Current min range:{min(scan_range) } ')
        done = not 0<min(scan_range) < 0.15 #Collision
        # boundary = math.sqrt(pow(self.position.x,2) + pow(self.position.y,2))
        # done=boundary>10

        current_distance = self.get_goal_distance(self.position)
        arrival = current_distance < (1.2 if ep < 30 else 0.7 if ep < 100 else 0.5)
        self.get_goalbox=current_distance < (1.2 if ep < 30 else 0.7 if ep < 100 else 0.5)
        
        
        return [self.heading, current_distance, obstacle_min_range, obstacle_angle], done, arrival
    def setReward(self, state, done, action):
    
        yaw_reward = []
        
        obstacle_angle = state[3]
        obstacle_min_range = state[2]
        current_distance = state[1]
        heading = state[0]      
        
        
        #angular_rate = abs(action[0])/1.5       
        #linear_rate  = action[1]/0.5
        
        #angular_reward = -2*angular_rate + 2
        #linear_reward  =  2*linear_rate
        #action_reward  = angular_reward + linear_reward
        
        if (obstacle_min_range < 0.7):
            obstacle_reward = -0.5/(obstacle_min_range+0.0001)
        else:
            obstacle_reward = 0.0
            
        lin_vel_rate = action[1]/0.5
        ang_vel_rate = abs(action[0])/1.5 
        lin_vel_reward = 3*lin_vel_rate
        ang_vel_reward = -2*ang_vel_rate+2
        action_reward = ang_vel_reward + lin_vel_reward
        
        distance_rate = (current_distance / (self.goal_distance+0.0000001))
        distance_reward = -2*distance_rate+2 if distance_rate<=1 else 1
        
        angle_rate = 2*abs(heading)/pi
        angle_reward = -3*angle_rate+3 if angle_rate<=1 else -1*angle_rate+1
        
        #time_reward = -self.time_step/20
        #time_reward = 1/(self.time_step +1)
        time_reward = -2
        
        reward = distance_reward * angle_reward + obstacle_reward + time_reward
        
        if not done:
            self.get_logger().warn('Collision!!!!')
            reward = -100
            self.time_step = 0
            self.pub_cmd_vel.publish(Twist())

        if self.get_goalbox:
            self.get_logger().info('Goal!!')
            reward = 500
            self.pub_cmd_vel.publish(Twist())
            self.delete = True
            #self.goal_x, self.goal_y = self.RespawnGoal.goal_def(self.delete)
            self.goal_x, self.goal_y = self.respawn_goal.get_position() #4.5, 4.5 # goal_def()
            print("NEXT GOAL : ", self.goal_x, self.goal_y )
            self.goal_distance = self.get_goal_distance(self.position)
            self.get_goalbox = False
            self.time_step = 0
            #time.sleep(0.2)
            
        #print("total Reward:%0.3f"%reward, "\n")
        #print("Reward : ", reward)
        
        return reward
    # def wait_for_message(self, topic, msg_type, timeout=None):
    #     future = self.create_subscription(msg_type, topic, lambda msg: future.set_result(msg), 10)
    #     return future.result(timeout=timeout)

    def reset(self,ep):
        self.get_logger().info(f'RESETTTTTT..................')
        while not self.reset_proxy.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.reset_proxy.call_async(Empty.Request())
        data=None
        while data is None:
            try:
                data = rclpy.wait_for_message.wait_for_message(LaserScan,self,'scan', qos_profile=10)
            except Exception as e:
                self.get_logger().warn(f"Failed to receive LaserScan message: {e}")
        self.goal_distance = self.get_goal_distance(self.position)
        state, done, _ = self.get_state(data[1], ep)
        return np.asarray(state)

def main():
    rclpy.init()
    env = Env()
    rclpy.spin(env)
    env.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
