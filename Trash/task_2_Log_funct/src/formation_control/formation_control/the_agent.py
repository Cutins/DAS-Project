import numpy as np
from time import sleep
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray as MsgFloat

class Agent(Node):
    def __init__(self):
        super().__init__('agent',
                        allow_undeclared_parameters=True,
                        automatically_declare_parameters_from_overrides=True)
        
        self.id = self.get_parameter('agent_id').value
        self.pos = np.array(self.get_parameter('pos_init').value)
        self.max_iters = self.get_parameter('max_iters').value
        self.distances = self.get_parameter('distances').value
        self.comm_time = self.get_parameter('comm_time').value
        self.euler_step = self.get_parameter('euler_step').value
        self.neighs = np.nonzero(self.distances)[0]
        self.kk = 0
        self.counter = 0
        
        # CREATE TOPIC
        self.publisher = self.create_publisher(msg_type=MsgFloat, 
                                                topic=f'/topic_{self.id}',
                                                qos_profile=10)

        # SUBSCRIBE TO NEIGHBORS
        for neigh in self.neighs:
            self.create_subscription(msg_type=MsgFloat,
                                    topic=f'/topic_{neigh}',
                                    callback=self.listener_callback,
                                    qos_profile=10)
            
        # CREATE TIMER FOR SYNCH
        self.timer = self.create_timer(timer_period_sec = self.comm_time,
                                       callback=self.timer_callback)

        # MASSAGES BUFFER 
        self.received_msgs = {neigh: [] for neigh in self.neighs}

        print(f"Setup of agent {self.id} completed.")


    def listener_callback(self, msg):
        id, kk, x, y, z = msg.data
        self.received_msgs[int(id)].append((int(kk), np.array([x, y, z])))


    def formation_dynamics(self):
        delta_pos = np.zeros((3,))
        for neigh in self.neighs:
            
            # Transform data to numpy array
            neigh_pos = self.received_msgs[neigh].pop(0)[1]
            neigh_dist = self.distances[neigh]

            # Formation Control Law
            formation_d_potential = (np.linalg.norm(self.pos - neigh_pos, ord=2)**2 - neigh_dist**2) * (self.pos - neigh_pos)# + (self.pos[2])
            barrier_d_potential = - (2* (self.pos - neigh_pos)/(np.linalg.norm(self.pos - neigh_pos, ord=2)**2))
            # barrier_d_potential = - neigh_dist * (2* (self.pos - neigh_pos)/(np.linalg.norm(self.pos - neigh_pos, ord=2)**2))

            # barrier_potential = -np.log(np.linalg.norm(self.pos - neigh_pos, ord=2)**2) 
            # if barrier_potential > 0:
            #     delta_pos = delta_pos - (formation_d_potential + barrier_d_potential)
            # else:
            #     delta_pos = delta_pos - formation_d_potential 

            delta_pos = delta_pos - (formation_d_potential + barrier_d_potential)

        return delta_pos


    def timer_callback(self):
        self.counter += 1
        if self.kk > 0: 

            all_received = all([self.received_msgs[neigh] != [] for neigh in self.neighs])
            if all_received:
                all_synch = all([self.received_msgs[neigh][0][0] == self.kk-1 for neigh in self.neighs])
                if all_synch:

                    # Dynamic Integration
                    delta_pos = self.formation_dynamics()

                    if self.id == 0:
                        delta_pos = np.array([0.01, 0, 0]) / self.euler_step
                    
                    # Update Agent Position and Euler Discretization
                    # self.pos += ((self.counter * self.integration_step)*0.1) * delta_pos
                    self.pos += self.euler_step * delta_pos
                    print(f'Counter:\n{self.counter}')
                    self.counter = 0

                    # Check Termination
                    if self.kk > self.max_iters:

                        print('\nMAX ITERATIONS')
                        sleep(3)
                        self.destroy_node()

                    msg = MsgFloat()
                    x, y, z = self.pos
                    msg.data = [float(self.id), float(self.kk), x, y, z]
                    self.publisher.publish(msg)

                    self.kk += 1

                    self.get_logger().info(f"Agent {int(msg.data[0]):d} -- Iter = {int(msg.data[1]):d}\n\tPosition:\n\t\tx: {msg.data[2]:.4f}\n\t\ty: {msg.data[3]:.4f}\n\t\tz: {msg.data[4]:.4f}")

        else: 
            msg = MsgFloat()
            x, y, z = self.pos
            msg.data = [float(self.id), float(self.kk), x, y, z]
            self.publisher.publish(msg)

            self.kk += 1
            self.get_logger().info(f"Agent {int(msg.data[0]):d} -- Iter = {int(msg.data[1]):d}\n\tPosition:\n\t\tx: {msg.data[2]:.4f}\n\t\ty: {msg.data[3]:.4f}\n\t\tz: {msg.data[4]:.4f}")
        


def main():
    rclpy.init()

    agent = Agent()
    agent.get_logger().info(f"Agent {agent.id:d} -- Waiting for sync...")
    sleep(1)
    agent.get_logger().info("GO!")

    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        agent.get_logger().info("----- Node stopped cleanly -----")
    finally:
        rclpy.shutdown() 


if __name__ == '__main__':
    main()   