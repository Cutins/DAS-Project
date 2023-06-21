import numpy as np
from time import sleep
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray as MsgFloat


def writer(file_name, string):
    """
      inner function for logging
    """
    file = open(file_name, "a") # "a" is for append
    file.write(string)
    file.close()


class Obstacle(Node):
    def __init__(self):
        super().__init__('obstacle',
                        allow_undeclared_parameters=True,
                        automatically_declare_parameters_from_overrides=True)
        
        self.id = self.get_parameter('obstacle_id').value
        self.pos = np.array(self.get_parameter('pos_init').value)
        self.comm_time = self.get_parameter('comm_time').value
        self.id_for_csv = (self.get_parameter('N_agents').value) + self.id
        

        # CREATE TOPIC
        self.publisher = self.create_publisher(msg_type=MsgFloat, 
                                                topic=f'/topic_obstacle_{self.id}',
                                                qos_profile=10)
            
        # CREATE TIMER FOR SYNCH
        self.timer = self.create_timer(timer_period_sec = self.comm_time,
                                       callback=self.timer_callback)
        
        ####### create logging file ######
        self.file_name = "_csv_file/agent_{}.csv".format(self.id_for_csv)
        file = open(self.file_name, "w+") # 'w+' needs to create file and open in writing mode if doesn't exist
        file.close()

        print(f"Setup of obstacle {self.id} completed.")


    def timer_callback(self):
        msg = MsgFloat()
        x, y, z = self.pos
        msg.data = [float(self.id), x, y, z]
        self.publisher.publish(msg)

        ######## LOG files #######
        # save on file
        data_for_csv = msg.data.tolist().copy()
        data_for_csv = [str(round(element,4)) for element in data_for_csv[1:3]]
        data_for_csv = ','.join(data_for_csv)
        writer(self.file_name,data_for_csv+'\n')
        # self.get_logger().info(f"Obstacle {int(msg.data[0]):d}\n\tPosition:\n\t\tx: {msg.data[1]:.4f}\n\t\ty: {msg.data[2]:.4f}\n\t\tz: {msg.data[3]:.4f}")
        

def main():
    rclpy.init()

    obstacles = Obstacle()

    try:
        rclpy.spin(obstacles)
    except KeyboardInterrupt:
        obstacles.get_logger().info("----- Node stopped cleanly -----")
    finally:
        rclpy.shutdown() 


if __name__ == '__main__':
    main()   