import rclpy
from rclpy.node import Node

from std_msgs.msg import String
# from .subscriber import MinimalSubscriber

class BridgePubSub(Node):

    def __init__(self):
        super().__init__('abcd')
        # timer_period = 0.5  # seconds
        # self.timer = self.create_timer(timer_period, self.listen_callback)
        self.subscription1 = self.create_subscription(
            String,
            'topic1',
            self.listener_callback1,
            10)
        self.subscription2 = self.create_subscription(
            String,
            'topic2',
            self.listener_callback2,
            10)
        self.subscription3 = self.create_subscription(
            String,
            'topic3',
            self.listener_callback3,
            10)
        self.subscription4 = self.create_subscription(
            String,
            'topic4',
            self.listener_callback4,
            10)
        self.topic1='Hello World'
        self.topic2='Hello World'
        self.topic3='Hello World'
        self.topic4='Hello World'
    def listener_callback1(self, msg):
        # self.get_logger().info('I heard: "%s"' % msg.data)
        self.topic1=msg.data
    
    def listener_callback2(self, msg):
        self.topic2=msg.data
    def listener_callback3(self, msg):
        self.topic3=msg.data
    def listener_callback4(self, msg):
        self.topic4=msg.data
    def listen_callback(self):
        self.get_logger().info(f'Bridge -> Topic 1: {self.topic1} \t Topic 2: {self.topic2} \t Topic 3: {self.topic3} \t Topic 4: {self.topic4}')

