import numpy as np
from time import sleep
import matplotlib.pyplot as plt
import os

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray as MsgFloat


############################  UTILS  ###################################
def writer(file_name, string):
    """
      inner function for logging
    """
    file = open(file_name, "a") # "a" is for append
    file.write(string)
    file.close()


############################# NODE #####################################
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
        self.amplitude = self.get_parameter('amplitude').value
        self.type_motion = self.get_parameter('type_motion').value
        self.neighs = np.nonzero(self.distances)[0]
        self.kk = 0
        self.counter = 0
        self.start_moving = 0
        
        # Obstacles
        self.N_obstacles = self.get_parameter('N_obstacles').value
        self.pos_obs = np.zeros((self.N_obstacles, self.pos.shape[0]))

        ####### create logging file ######
        self.file_name = "_csv_file/agent_{}.csv".format(self.id)
        folder_path = os.path.dirname(self.file_name)
        if os.path.exists(folder_path) == False:
            os.makedirs(folder_path)
        file = open(self.file_name, "w+")
        file.close()

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
        self.start_moving = msg.data[0]
        self.pos_target = np.array([msg.data[1 + self.id*3 + idx] for idx in range(3)])
        self.k_start_moving = self.kk


    # OBSTACLE
    def listener_callback_obstacle(self, msg):
        id, x, y, z = msg.data
        self.pos_obs[int(id)] = [x, y, z]


    def timer_callback(self):
        self.counter += 1
        if self.kk > 0: 

            all_received = all([self.received_msgs[neigh] != [] for neigh in self.neighs])
            if all_received:

                all_synch = all([self.received_msgs[neigh][0][0] == self.kk-1 for neigh in self.neighs])
                if all_synch:

                    # Formation
                    delta_pos = self.formation_dynamics()

                    # Moving formation
                    if self.start_moving and self.type: # Leader
                        # circular
                        if self.type_motion == 0: 
                            delta_pos = delta_pos + self.circle_trajectory(frequency=1, phase=0)

                        # target position
                        elif self.type_motion == 1: 
                            delta_pos = delta_pos + self.target_trajectory(gain=1)

                        # linear
                        elif self.type_motion == 2: 
                            delta_pos = delta_pos + self.linear_trajectory()

                        else:
                            assert 'Type of Motion not available!'

                    if self.start_moving and (self.type == 0): # Follower
                        delta_pos = delta_pos + self.obstacle_avoidance()

                    # Update of dynamics
                    self.pos += self.euler_step * delta_pos

                    # Print number of iteration for synch
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
                    
                    # save on file
                    data_for_csv = msg.data.tolist().copy()
                    data_for_csv = [str(round(element,4)) for element in data_for_csv[2:4]]
                    data_for_csv = ','.join(data_for_csv)
                    writer(self.file_name,data_for_csv+'\n')

        else: 
            msg = MsgFloat()
            x, y, z = self.pos
            msg.data = [float(self.id), float(self.kk), x, y, z]
            self.publisher.publish(msg)

            self.kk += 1
            
            self.get_logger().info(f"Agent {int(msg.data[0]):d} -- Iter = {int(msg.data[1]):d}\n\tPosition:\n\t\tx: {msg.data[2]:.4f}\n\t\ty: {msg.data[3]:.4f}\n\t\tz: {msg.data[4]:.4f}")
        
            # save on file
            data_for_csv = msg.data.tolist().copy()
            data_for_csv = [str(round(element,4)) for element in data_for_csv[2:4]]
            data_for_csv = ','.join(data_for_csv)
            writer(self.file_name,data_for_csv+'\n')


    # DYNAMICS AND TRAJECTORIES
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


    def circle_trajectory(self, frequency=1, phase=0):
        time = self.kk - self.k_start_moving
        omega_n = (2*np.pi) / (self.max_iters - self.k_start_moving)
        x = self.amplitude * np.cos((frequency * omega_n * time) + phase)
        y = self.amplitude * np.sin((frequency * omega_n * time) + phase)
        z = 0
        gradient = np.array([x, y, z])
        return gradient


    def linear_trajectory(self):
        x = 0
        y = self.amplitude
        z = 0
        gradient = np.array([x, y, z])
        return gradient


    def target_trajectory(self, gain=1):
        gradient = gain * (self.pos_target - self.pos)
        return gradient



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