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
        self.type = np.array(self.get_parameter('type').value) # Leader = 1
        # self.input = np.array(self.get_parameter('input').value)
        self.move = self.get_parameter('move').value
        self.neighs = np.nonzero(self.distances)[0]
        self.kk = 0
        self.counter = 0
        self.start_moving = 0
        
        # For obstacle avoidance
        self.N_obstacles = self.get_parameter('N_obstacles').value
        self.pos_obs = np.zeros((self.N_obstacles, self.pos.shape[0]))

        # CREATE TOPIC
        self.publisher = self.create_publisher(msg_type=MsgFloat, 
                                                topic=f'/topic_{self.id}',
                                                qos_profile=10)

        # SUBSCRIBE TO NEIGHBORS and MANAGER and OBSTACLE
        for neigh in self.neighs:
            self.create_subscription(msg_type=MsgFloat,
                                    topic=f'/topic_{neigh}',
                                    callback=self.listener_callback,
                                    qos_profile=10)
            
        self.create_subscription(msg_type=MsgFloat,
                                topic='/topic_Manager',
                                callback=self.listener_callback_manager,
                                qos_profile=10)
        
        for obstacle in range(self.N_obstacles):
            self.create_subscription(msg_type=MsgFloat,
                                    topic=f'/topic_obstacle_{obstacle}',
                                    callback=self.listener_callback_obstacle,
                                    qos_profile=10)
            
        # CREATE TIMER FOR SYNCH
        self.timer = self.create_timer(timer_period_sec = self.comm_time,
                                       callback=self.timer_callback)

        # MASSAGES BUFFER 
        self.received_msgs = {neigh: [] for neigh in self.neighs}

        print(f"Setup of agent {self.id} completed.")


    # NEIGHBORS
    def listener_callback(self, msg):
        id, kk, x, y, z = msg.data
        self.received_msgs[int(id)].append((int(kk), np.array([x, y, z])))

    # MANAGER
    def listener_callback_manager(self, msg):
        self.start_moving = msg.data
        self.k_start_moving = self.kk

    # OBSTACLE
    def listener_callback_obstacle(self, msg):
        id, x, y, z = msg.data
        self.pos_obs[int(id)] = [x, y, z]



    def formation_dynamics(self):
        delta_pos = np.zeros((3,))
        for neigh in self.neighs:
            
            # Transform data to numpy array
            neigh_pos = self.received_msgs[neigh].pop(0)[1]
            neigh_dist = self.distances[neigh]

            # Formation Control Law
            formation_d_potential = (np.linalg.norm(self.pos - neigh_pos, ord=2)**2 - neigh_dist**2) * (self.pos - neigh_pos)
            barrier_d_potential = - (2* (self.pos - neigh_pos)/(np.linalg.norm(self.pos - neigh_pos, ord=2)**2))

            delta_pos = delta_pos - (formation_d_potential + barrier_d_potential)

        return delta_pos
    
    def obstacle_avoidance(self):
        delta_pos = np.zeros((3,))
        for obstacle in range(self.N_obstacles):
            obstacle_d_potential = - (2* (self.pos - self.pos_obs[obstacle])/(np.linalg.norm(self.pos - self.pos_obs[obstacle], ord=2)**2))
            delta_pos = delta_pos - (obstacle_d_potential)
        
        return delta_pos

    def circle_trajectory (self, time, amplitude, frequency = 1, phase = 0):
        omega_n = ((2*np.pi)/(self.max_iters - self.k_start_moving))
        x = amplitude * np.cos((frequency *omega_n *time) + phase)
        y = amplitude * np.sin((frequency *omega_n *time) + phase)
        z = 0

        gradient = np.array([x, y, z])
        return gradient


    def waves(self, amp, omega, phi, t):
        """
        This function generates a sinusoidal input trajectory

        Input:
            - amp, omega, phi = sine parameter u = amp*sin(omega*t + phi)
            - n_agents = number of agents
            - n_x = agent dimension
            - t = time variable

        Output:
            - u = input trajectory

        """

        u_x = amp*np.sin(omega*t+phi)
        u_y = 0
        u_z = 0
        u = [u_x, u_y, u_z]
        return np.array(u)


    def linear_trajectory (self, amplitude):
        x = 0
        y = amplitude
        z = 0

        gradient = np.array([x, y, z])
        return gradient

    def timer_callback(self):
        self.counter += 1
        if self.kk > 0: 

            all_received = all([self.received_msgs[neigh] != [] for neigh in self.neighs])
            if all_received:
                all_synch = all([self.received_msgs[neigh][0][0] == self.kk-1 for neigh in self.neighs])
                if all_synch:

                    # Formation
                    delta_pos = self.formation_dynamics()

                    # Leaders moving
                    if self.start_moving and self.type: # Leader
                        # discrete_time = self.kk - self.k_start_moving
                        # delta_pos = delta_pos + self.waves(amp= self.input, omega= omega_n, phi= 0, t= discrete_time)
                        # delta_pos = delta_pos + (self.circle_trajectory(discrete_time, self.move))
                        delta_pos = delta_pos + self.linear_trajectory(self.move)
                        # delta_pos = delta_pos + self.obstacle_avoidance()

                    if self.start_moving and (self.type == 0): # Follower
                        delta_pos = delta_pos + self.obstacle_avoidance()

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