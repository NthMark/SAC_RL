import rclpy
from rclpy.node import Node
from csv import writer
from std_msgs.msg import Float32MultiArray
from glob import glob

class Analysis(Node):

    def __init__(self):
        super().__init__('analysis_node')
        self.goal_distance_and_heading_subs = self.create_subscription(
            Float32MultiArray,
            'goal_distance_and_heading',
            self.goal_distance_and_heading_callback,
            10)
        self.minimum_distance_from_obstacles_subs = self.create_subscription(
            Float32MultiArray,
            'minimum_distance_obstacle',
            self.minimum_distance_from_obstacles_callback,
            10)
        self.velocity_subs = self.create_subscription(
            Float32MultiArray,
            'vel_output',
            self.velocity_callback,
            10)
        self.loss_subs = self.create_subscription(
            Float32MultiArray,
            'loss',
            self.loss_callback,
            10)
        self.entropy_subs = self.create_subscription(
            Float32MultiArray,
            'entropy',
            self.entropy_callback,
            10)
        
        self.goal_distance_and_heading_new_file_idx=1
        self.minimum_distance_from_obstacles_new_file_idx=1
        self.velocity_new_file_idx=1
        self.loss_new_file_idx=1
        self.entropy_new_file_idx=1

        self.goal_distance_and_heading_no_rows=0
        self.minimum_distance_from_obstacles_no_rows=0
        self.velocity_no_rows=0
        self.loss_no_rows=0
        self.entropy_no_rows=0
        self.maximum_rows=1000000
        self.no_idx=len(glob('/home/mark/limo_ws/src/rl_sac/rl_sac/analysis/loss*'))

        while self.no_idx >0:
            with open(f'/home/mark/limo_ws/src/rl_sac/rl_sac/analysis/goal_distance_and_heading{self.no_idx}.csv', 'w', newline='') as csvfile:
                pass
            with open(f'/home/mark/limo_ws/src/rl_sac/rl_sac/analysis/entropy{self.no_idx}.csv', 'w', newline='') as csvfile:
                pass
            with open(f'/home/mark/limo_ws/src/rl_sac/rl_sac/analysis/loss{self.no_idx}.csv', 'w', newline='') as csvfile:
                pass
            with open(f'/home/mark/limo_ws/src/rl_sac/rl_sac/analysis/minimum_distance_from_obstacles{self.no_idx}.csv', 'w', newline='') as csvfile:
                pass
            with open(f'/home/mark/limo_ws/src/rl_sac/rl_sac/analysis/velocity{self.no_idx}.csv', 'w', newline='') as csvfile:
                pass
            self.no_idx-=1
    def goal_distance_and_heading_callback(self,msg):
        with open(f'/home/mark/limo_ws/src/rl_sac/rl_sac/analysis/goal_distance_and_heading{self.goal_distance_and_heading_new_file_idx}.csv', 'a', newline='') as csvfile:
            self.goal_distance_and_heading_no_rows+=1
            if self.goal_distance_and_heading_no_rows ==self.maximum_rows:
                self.goal_distance_and_heading_no_rows=0
                self.goal_distance_and_heading_new_file_idx+=1
            csvwriter = writer(csvfile, delimiter='|')
            csvwriter.writerow(msg.data)
        
    def minimum_distance_from_obstacles_callback(self,msg):
        with open(f'/home/mark/limo_ws/src/rl_sac/rl_sac/analysis/minimum_distance_from_obstacles{self.minimum_distance_from_obstacles_new_file_idx}.csv', 'a', newline='') as csvfile:
            self.minimum_distance_from_obstacles_no_rows+=1
            if self.minimum_distance_from_obstacles_no_rows ==self.maximum_rows:
                self.minimum_distance_from_obstacles_no_rows=0
                self.minimum_distance_from_obstacles_new_file_idx+=1
            csvwriter = writer(csvfile, delimiter='|')
            csvwriter.writerow(msg.data)

    def velocity_callback(self,msg):
        with open(f'/home/mark/limo_ws/src/rl_sac/rl_sac/analysis/velocity{self.velocity_new_file_idx}.csv', 'a', newline='') as csvfile:
            self.velocity_no_rows+=1
            if self.velocity_no_rows ==self.maximum_rows:
                self.velocity_no_rows=0
                self.velocity_new_file_idx+=1
            csvwriter = writer(csvfile, delimiter='|')
            csvwriter.writerow(msg.data)

    def loss_callback(self,msg):
        with open(f'/home/mark/limo_ws/src/rl_sac/rl_sac/analysis/loss{self.loss_new_file_idx}.csv', 'a', newline='') as csvfile:
            self.loss_no_rows+=1
            if self.loss_no_rows ==self.maximum_rows:
                self.loss_no_rows=0
                self.loss_new_file_idx+=1
            csvwriter = writer(csvfile, delimiter='|')
            csvwriter.writerow(msg.data)
            
    def entropy_callback(self,msg):
        with open(f'/home/mark/limo_ws/src/rl_sac/rl_sac/analysis/entropy{self.entropy_new_file_idx}.csv', 'a', newline='') as csvfile:
            self.entropy_no_rows+=1
            if self.entropy_no_rows ==self.maximum_rows:
                self.entropy_no_rows=0
                self.entropy_new_file_idx+=1
            csvwriter = writer(csvfile, delimiter='|')
            csvwriter.writerow(msg.data)


def main(args=None):
    rclpy.init(args=args)

    analysis_node = Analysis()

    rclpy.spin(analysis_node)

    analysis_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()