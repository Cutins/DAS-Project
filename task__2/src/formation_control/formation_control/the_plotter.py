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
        self.distance_matrix = self.get_parameter('distance_matrix').value
        self.comm_time = self.get_parameter('comm_time').value
        self.n_agents = self.distance_matrix.shape[0]
        self.pos = np.zeros((self.max_iters, self.n_agents, 3))
        self.potential = np.zeros((self.max_iters, self.n_agents))
        self.kk = 0

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


    def compute_dynamics(self, data):
        delta_pos = np.zeros((3,))
        for neigh in data.keys():
            
            # Transform data to numpy array
            neigh_pos = data[neigh]['pos']
            neigh_dist = data[neigh]['dist']

            # Formation Control Law
            formation_potential = (np.linalg.norm(self.pos - neigh_pos, ord=2)**2 - neigh_dist**2) * (self.pos - neigh_pos)# + (self.pos[2])
            barrier_potential = - 2* (self.pos - neigh_pos)/(np.linalg.norm(self.pos - neigh_pos, ord=2)**2)
            
            delta_pos = delta_pos - (formation_potential + barrier_potential)
            #delta_pos = delta_pos - formation_potential 

        return delta_pos

    def compute_potential(self):

        for agent in range(self.n_agents):
            self.pos[agent] = self.received_msgs[agent].pop(0)[1]
            

    
        return
    
    def timer_callback(self):
        all_received = all([self.received_msgs[agent] != [] for agent in range(self.n_agents)])

        if all_received: 
            all_synch = all([self.received_msgs[agent][0][0] == self.kk for agent in range(self.n_agents)])
            
            if all_synch:       
    
                for agent in range(self.n_agents):
                    self.pos[self.kk, agent] = self.received_msgs[agent].pop(0)[1]
                    neighs = np.nonzero(self.distance_matrix[agent])[0]
                    self.potential[self.kk, agent] = np.sum([1/4*(np.linalg.norm(self.pos[agent] - self.pos[neigh], ord=2)**2 - self.distance_matrix[agent][neigh]**2)**2 for neigh in neighs])


                if self.kk == self.max_iters:
                    for agent in range(self.n_agents):
                        plt.plot(range(self.max_iters), self.potential[:,agent], label=f'Potential of agent {agent}')
                        
                    plt.label()
                    plt.grid()
                    plt.show()
                        

                # Check Termination
                if self.kk > self.max_iters:

                    print('\nMAX ITERATIONS')
                    sleep(3)
                    self.destroy_node()

                self.kk += 1
                


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