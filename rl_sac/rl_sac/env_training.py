import copy
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
    def __init__(self,action_dim=2):
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
        self.stopped = 0
        self.action_dim = action_dim
        self.reset_proxy = self.create_client(Empty, 'reset_simulation')
        self.unpause_proxy = self.create_client(Empty, 'unpause_physics')
        self.pause_proxy = self.create_client(Empty, 'pause_physics')
        self.past_distance = 0.
        self.pub_cmd_vel = self.create_publisher(Twist, 'cmd_vel', 10)
        # self.sub_odom = self.create_subscription(Odometry, 'odom', self.get_odometry, 10) 
        # self.sub_scan = self.create_subscription(LaserScan, 'scan', self.get_scan_data, 10)
        self.scan = LaserScan()
    # def get_scan_data(self, scan_msg:LaserScan):
    #     # self.get_logger.info(scan_msg)
    #     self.scan = scan_msg
    #     for i in range(len(self.scan.ranges)):
    #         if self.scan.ranges[i] == float('Inf') or self.scan.ranges[i] == float('inf'):
    #             pass
    #         elif np.isnan(self.scan.ranges[i]) or self.scan.ranges[i] == float('nan'):
    #             pass
    #         else:
    #             self.get_logger().info(f"{i}-------{self.scan.ranges[i]}")
        # if len(self.scan.ranges) > 0:
        #     self.get_logger().info(f"--------------------{self.scan.ranges}")

    def get_goal_distance(self,position):
        goal_distance = round(math.hypot(self.goal_x - position.x, self.goal_y - position.y), 4)
        self.past_distance = goal_distance

        return goal_distance

    # def get_odometry(self, odom):
    #     self.past_position = copy.deepcopy(self.position)
    #     self.position = odom.pose.pose.position
    #     orientation = odom.pose.pose.orientation
    #     orientation_list = [orientation.w,orientation.x, orientation.y, orientation.z]
    #     _, _, yaw = quat2euler(orientation_list)

    #     goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)
    #     self.heading = goal_angle - yaw
    #     if self.heading > pi:
    #         self.heading -= 2 * pi
    #     elif self.heading < -pi:
    #         self.heading += 2 * pi
    def step(self, action, past_action,v_max,w_max):
        
        self.time_step += 1
        
        vel_cmd = Twist()
        vel_cmd.linear.x  =  float(action[1]) 
        vel_cmd.angular.z = float(action[0]) 
        self.pub_cmd_vel.publish(vel_cmd)
        
        #print("EP:", ep, " Step:", t, " Goal_x:",self.goal_x, "  Goal_y:",self.goal_y)
        data=None
        while data is None:
            try:
                data = rclpy.wait_for_message.wait_for_message(LaserScan,self,'scan', qos_profile=10) #(boolean,message)
            except Exception as e:
                self.get_logger().warn(f"Failed to receive LaserScan message: {e}")
        state, done,min_distance_obstacle,reach_goal = self.get_state(data[1], past_action)
        reward,done = self.setReward(state, done, action,v_max,w_max)
        return np.asarray(state), reward, done ,min_distance_obstacle,reach_goal
    def get_state(self, scan, past_action):
        ###########
        odom=None
        while odom is None:
            try:
                odom = rclpy.wait_for_message.wait_for_message(Odometry,self,'odom', qos_profile=10) #(boolean,message)
            except Exception as e:
                self.get_logger().warn(f"Failed to receive LaserScan message: {e}")
        self.past_position = copy.deepcopy(self.position)
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
        raw_scan_range=[scan.ranges[0],scan.ranges[20],scan.ranges[40],scan.ranges[60],scan.ranges[80],scan.ranges[90],
                  scan.ranges[350],scan.ranges[340],scan.ranges[320],scan.ranges[300],scan.ranges[280],scan.ranges[270]]
        
        scan_range=[3.5 if r==float('Inf') or r==float('inf') else 0 if np.isnan(r) or r==float('nan') else r for r in raw_scan_range]
        # obstacle_min_range = round(min(scan_range), 2)
        # obstacle_angle = np.argmin(scan_range)
        # self.get_logger().info(f'Current pose: \n x:{self.position.x} y:{self.position.y}')
        # self.get_logger().info(f'Current min range:{min(scan_range) } ')
        
        done =  0<min(scan.ranges) < 0.15 #Collision
        # boundary = math.sqrt(pow(self.position.x,2) + pow(self.position.y,2))
        # done=boundary>10

        current_distance = self.get_goal_distance(self.position)
        self.get_logger().info(f'Current min range:{current_distance }')
        # arrival = current_distance < (1.2 if ep < 30 else 0.7 if ep < 100 else 0.5)
        self.get_goalbox=current_distance < 0.15
        
        
        return scan_range + list(past_action)+[self.heading, current_distance], done,min(scan.ranges),self.get_goalbox
    def setReward(self, state, done, action,v_max,w_max):
    
        # yaw_reward = []
        
        # obstacle_angle = state[3]
        # obstacle_min_range = state[2]
        current_distance = state[-1]
        heading = state[-2]      
        ########Case 1###########
        distance_rate = (self.past_distance - current_distance) 
        if distance_rate > 0:
            # reward = 200.*distance_rate
            reward = 0.

        # if distance_rate == 0:
        #     reward = 0.

        if distance_rate <= 0:
            # reward = -8.
            reward = 0.
        ########Case 1###########
        
        ########Case 2###########
        # lin_vel_rate = action[1]/v_max
        # ang_vel_rate = abs(action[0])/w_max
        # lin_vel_reward = 3*lin_vel_rate
        # ang_vel_reward = -2*ang_vel_rate+2
        # action_reward = ang_vel_reward + lin_vel_reward
        
        # distance_rate = (current_distance / (self.goal_distance+0.0000001))
        # distance_reward = -2*distance_rate+2 if distance_rate<=1 else 1
        
        # angle_rate = 2*abs(heading)/pi
        # angle_reward = -3*angle_rate+3 if angle_rate<=1 else -1*angle_rate+1
        
        # #time_reward = -self.time_step/20
        # #time_reward = 1/(self.time_step +1)
        # time_reward = -2
        # reward = distance_reward * angle_reward 
        ########Case 2###########
        a, b, c, d = float('{0:.3f}'.format(self.position.x)), float('{0:.3f}'.format(self.past_position.x)), float('{0:.3f}'.format(self.position.y)), float('{0:.3f}'.format(self.past_position.y))
        if a == b and c == d:
            self.stopped += 1
            if self.stopped == 20:
                self.get_logger().warn('Robot is in the same 20 times in a row')
                self.stopped = 0
                done = True
        else:
            self.stopped = 0

        if done:
            self.get_logger().warn('Collision!!!!')
            # reward = -100
            reward = -30
            self.time_step = 0
            self.pub_cmd_vel.publish(Twist())

        if self.get_goalbox:
            self.get_logger().info('Goal!!')
            # reward = 500
            reward = 100
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
        
        return reward,done
    # def wait_for_message(self, topic, msg_type, timeout=None):
    #     future = self.create_subscription(msg_type, topic, lambda msg: future.set_result(msg), 10)
    #     return future.result(timeout=timeout)

    def reset(self):
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
        state, _ ,min_distance_obstacle,reach_goal= self.get_state(data[1], [0]*self.action_dim)
        return np.asarray(state),min_distance_obstacle,reach_goal

def main():
    rclpy.init()
    env = Env()
    rclpy.spin(env)
    env.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
