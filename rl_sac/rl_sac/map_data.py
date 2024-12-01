import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid,Odometry
class MapSubscriber(Node):

    def __init__(self):
        super().__init__('map_data')
        self.subscription1 = self.create_subscription(
            Odometry,
            'odom',
            self.listener_callback1,
            10)
        self.i=0
    def listener_callback1(self,msg):
        self.i+=1
        if self.i==10:
            self.get_logger().info(f"Current position: {msg.pose.pose}")
            self.i=0
        

def main(args=None):
    rclpy.init(args=args)

    map_subscriber = MapSubscriber()

    rclpy.spin(map_subscriber)
    map_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()