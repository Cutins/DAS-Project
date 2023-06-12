import numpy as np
from time import sleep
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray as MsgFloat

class Plot(Node):
    def __init__(self):
        super().__init__('plot',
                        allow_undeclared_parameters=True,
                        automatically_declare_parameters_from_overrides=True)
        
        self.max_iters = self.get_parameter('max_iters').value
        self.n_agents = np.sqrt(len(self.get_parameter('distance_matrix').value)).astype(int)
        self.distance_matrix = np.array(self.get_parameter('distance_matrix').value).reshape((self.n_agents, self.n_agents))
        self.comm_time = self.get_parameter('comm_time').value
        print(f'Distance matrix:\n{self.distance_matrix}')
        
        # self.n_agents = self.distance_matrix.shape[0]
        self.pos = np.zeros((self.max_iters, self.n_agents, 3))
        self.potential = np.zeros((self.max_iters, self.n_agents))
        self.formation =  np.zeros((self.max_iters, self.n_agents))
        self.all_in_formation = np.zeros((self.max_iters))
        self.start_moving = 0
        self.kk = 0

        # Create a Topic to comunicate at the agent that they're in formation
        self.publisher = self.create_publisher(msg_type=MsgFloat, 
                                                topic='/topic_Manager',
                                                qos_profile=10)


        # SUBSCRIBE TO NEIGHBORS
        for agent in range(self.n_agents):
            self.create_subscription(msg_type=MsgFloat,
                                    topic=f'/topic_{agent}',
                                    callback=self.listener_callback,
                                    qos_profile=100)
            
        # CREATE TIMER FOR SYNCH
        self.timer = self.create_timer(timer_period_sec = self.comm_time,
                                       callback=self.timer_callback)

        # MASSAGES BUFFER 
        self.received_msgs = {agent: [] for agent in range(self.n_agents)}

        print(f"Setup of plot completed.")


    def listener_callback(self, msg):
        id, kk, x, y, z = msg.data
        self.received_msgs[int(id)].append((int(kk), np.array([x, y, z])))

    
    def timer_callback(self):
        
        all_received = all([self.received_msgs[agent] != [] for agent in range(self.n_agents)])
        if all_received: 

            all_synch = all([self.received_msgs[agent][0][0] == self.kk for agent in range(self.n_agents)])
            if all_synch:      

                for agent in range(self.n_agents):
                    self.pos[self.kk, agent] = self.received_msgs[agent].pop(0)[1]
                for agent in range(self.n_agents):
                    neighs = np.nonzero(self.distance_matrix[agent])[0]
                    self.potential[self.kk, agent] = np.sum([1/4*(np.linalg.norm(self.pos[self.kk, agent] - self.pos[self.kk, neigh], ord=2)**2 - self.distance_matrix[agent][neigh]**2)**2 for neigh in neighs])
                
                                # Switch to moving leaders
                if self.kk > 5:
                    # Check if derivative is equal to zero
                    for agent in range(self.n_agents):
                        if np.abs(self.potential[self.kk-5, agent]-self.potential[self.kk, agent]) < 0.5e-1:
                            self.formation[self.kk, agent] = 1
                    # If all agents have the derivative equal to zero
                    if np.all(self.formation[self.kk]):
                        self.all_in_formation[self.kk] = 50
                        if self.start_moving == 0: # Detect rising edge, first time we reach formation 
                            print("\n\n-------FORMATION-------\n\n")
                            self.start_moving = 1.0
                            msg = MsgFloat()
                            msg.data = [self.start_moving]
                            self.publisher.publish(msg)

                self.kk += 1

                # Check Termination
                if self.kk == self.max_iters:
                    print('\nSTART PLOTTING')
                    
                                            
                    plt.figure('Potential')
                    for agent in range(self.n_agents):
                        plt.plot(range(self.max_iters), self.potential[:,agent], ':', label=f'Potential of agent {agent}') 
                    plt.plot(range(self.max_iters), np.mean(self.potential, axis=-1), label=f'Formation potential', linewidth = 2) 
                    plt.legend()
                    plt.grid()

                    plt.show()
                        
                if self.kk >= self.max_iters:

                    print('\nMAX ITERATIONS')
                    sleep(3)
                    self.destroy_node()

                


def main():
    rclpy.init()

    plots = Plot()

    try:
        rclpy.spin(plots)
    except KeyboardInterrupt:
        print("----- Node stopped cleanly -----")
    finally:
        rclpy.shutdown() 


if __name__ == '__main__':
    main()   