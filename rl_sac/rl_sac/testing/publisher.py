import rclpy
from rclpy.node import Node
import time
from std_msgs.msg import String


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_1 = self.create_publisher(String, 'topic1', 10)
        self.publisher_2 = self.create_publisher(String, 'topic2', 10)
        self.publisher_3 = self.create_publisher(String, 'topic3', 10)
        self.publisher_4 = self.create_publisher(String, 'topic4', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        j=0
        msg = String()
        msg.data = 'Hello World 111: %d' % self.i
        while j<10:
            time.sleep(0.5)
            j+=1
            self.get_logger().info(f'Publishing {j} in {self.i}')
        self.publisher_1.publish(msg)
        msg.data = 'Hello World 222: %d' % self.i
        self.publisher_2.publish(msg)
        msg.data = 'Hello World 333: %d' % self.i
        self.publisher_3.publish(msg)
        msg.data = 'Hello World 444: %d' % self.i
        self.publisher_4.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()