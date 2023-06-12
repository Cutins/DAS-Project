import numpy as np
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose
from std_msgs.msg import Float32MultiArray as MsgFloat

class Visualizer_obstacle(Node):

    def __init__(self):
        super().__init__('visualizer_obstacle',
                            allow_undeclared_parameters=True,
                            automatically_declare_parameters_from_overrides=True)
        
        # Get parameters from launcher
        self.agent_id = self.get_parameter('agent_id').value
        self.comm_time = self.get_parameter('comm_time').value


        #######################################################################################
        # Let's subscribe to the topic we want to visualize
        
        self.subscription = self.create_subscription(MsgFloat, 
                                                     f'/topic_obstacle_{self.agent_id}',
                                                     self.listener_callback, 
                                                     qos_profile=10)

        #######################################################################################

        # Create the publisher that will communicate with Rviz
        self.timer = self.create_timer(timer_period_sec=self.comm_time, callback=self.publish_data)
        self.publisher = self.create_publisher(Marker, 
                                               f'/visualization_topic_obstacle', 
                                               qos_profile=1)

        # Initialize the current_pose method (in this example you can also use list or np.array)                                         
        self.current_pose = Pose()


    def listener_callback(self, msg):
        # store (and rearrange) the received message
        self.current_pose.position.x = msg.data[1]
        self.current_pose.position.y = msg.data[2]
        self.current_pose.position.z = msg.data[3]
            
    def publish_data(self):
        if self.current_pose.position is not None:
            # Set the type of message to send to Rviz -> Marker
            # (see http://docs.ros.org/en/noetic/api/visualization_msgs/html/index-msg.html)
            marker = Marker()

            # Select the name of the reference frame, without it markers will be not visualized
            marker.header.frame_id = 'my_frame'
            marker.header.stamp = self.get_clock().now().to_msg()

            # Select the type of marker
            marker.type = Marker.CUBE

            # set the pose of the marker (orientation is omitted in this example)
            marker.pose.position.x = self.current_pose.position.x
            marker.pose.position.y = self.current_pose.position.y
            marker.pose.position.z = self.current_pose.position.z 

            # Select the marker action (ADD, DELATE)
            marker.action = Marker.ADD

            # Select the namespace of the marker
            marker.ns = 'obstacle'

            # Let the marker be unique by setting its id
            marker.id = self.agent_id

            # Specify the scale of the marker
            scale = 0.2
            marker.scale.x = scale
            marker.scale.y = scale
            marker.scale.z = scale

            # Specify the color of the marker as RGBA
            # if self.agent_id == 0:
            #     color = [1.0, 0.0, 0.0, 1.0]   # Red - Follower
            # if self.agent_id == 1:
            #     color = [0.0, 0.5, 0.5, 1.0]    # Blue - Leader

            color = [0.0, 1.0, 0.0, 1.0]   # Green


            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = color[3]

            # Let's publish the marker
            self.publisher.publish(marker)

def main():
    rclpy.init()

    visualizer_obstacle = Visualizer_obstacle()

    try:
        rclpy.spin(visualizer_obstacle)
    except KeyboardInterrupt:
        print("----- Visualizer_obstacle stopped cleanly -----")
    finally:
        rclpy.shutdown() 

if __name__ == '__main__':
    main()