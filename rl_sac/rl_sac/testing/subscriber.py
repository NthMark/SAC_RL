import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from .bridge_pub_sub import BridgePubSub
from rclpy.executors import SingleThreadedExecutor,MultiThreadedExecutor
class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('asdfdf')
        self.bridge_pub_sub = BridgePubSub()  

        timer_period=0.5
        self.timer=self.create_timer(timer_period,self.test_multiple_subs)
    def test_multiple_subs(self):
        self.bridge_pub_sub.listen_callback()

def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()
    executor= SingleThreadedExecutor()
    executor.add_node(minimal_subscriber)
    executor.add_node(minimal_subscriber.bridge_pub_sub)
    executor.spin()
    # rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()