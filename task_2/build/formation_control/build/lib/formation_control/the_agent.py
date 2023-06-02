import numpy as np
from time import sleep

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray as MsgFloat

class Agent(Node):
    def __init__(self):
        super().__init__('agent',
                        allow_undeclared_parameters=True,
                        automatically_declare_parameters_from_overrides=True)
        
        self.id = self.get_parameter('agent_id').value
        self.pos = tuple(self.get_parameter('pos_init').value)
        self.max_iters = self.get_parameter('max_iters').value
        self.distances = self.get_parameter('distances').value
        self.integration_step = self.get_parameter('integration_step').value
        self.neighs = np.nonzero(self.distances)[0]
        self.kk = 0

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
        timer_period = 1e-1
        self.timer = self.create_timer(timer_period_sec=timer_period,
                                       callback=self.timer_callback)

        # MASSAGES BUFFER 
        self.received_msgs = {neigh: [] for neigh in self.neighs}

        print(f"Setup of agent {self.id} completed.")


    def listener_callback(self, msg):
        id, kk, x, y, z = msg.data
        self.received_msgs[int(id)].append((int(kk), (x, y, z)))

    
    def compute_dynamics(self, data, integration_step):
        delta_pos = np.zeros((3,))
        for neigh in data.keys():
            
            # Transform data to numpy array
            neigh_pos = np.array(data[neigh]['pos'])
            neigh_dist = np.array(data[neigh]['dist'])

            # Formation Control Law
            formation_potential = np.linalg.norm((self.pos - neigh_pos)**2 - neigh_dist)**2
            randevouz_potential = self.pos - neigh_pos
            barrier_potential = -0
            delta_pos = delta_pos - (formation_potential * randevouz_potential + barrier_potential)
        
        # Euler Discretization
        delta_pos = integration_step * delta_pos

        return delta_pos


    def timer_callback(self):
        if self.kk > 0:
            all_received = all([self.received_msgs[neigh][0][0] == self.kk-1 for neigh in self.neighs])
            if all_received:

                # Dynamic Integration
                delta_pos = self.compute_dynamics({neigh: 
                                                        {
                                                        'pos': self.received_msgs[neigh][0][1],
                                                        'dist': self.distances[self.id][neigh]
                                                        }
                                                    for neigh in self.neighs})
                
                # Update Agent Position
                self.pos += delta_pos

                # Check Termination
                if self.kk > self.max_iters:
                    print('\nMAX ITERATIONS')
                    sleep(3)
                    self.destroy_node()

        msg = MsgFloat()
        x, y, z = self.pos
        msg.data = [float(self.id), float(self.kk), x, y, z]
        self.publisher.publish(msg)

        self.get_logger().info(f"Agent {int(msg.data[0]):d} -- Iter = {int(msg.data[1]):d}\n\tPosition:\n\t\tx: {msg.data[2]:.4f}\n\t\ty: {msg.data[3]:.4f}\n\t\tz: {msg.data[4]:.4f}")
        self.kk += 1


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