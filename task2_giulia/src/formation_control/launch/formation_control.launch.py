from launch import LaunchDescription
from launch_ros.actions import Node
import numpy as np
import networkx as nx
import os
from ament_index_python.packages import get_package_share_directory

MAXITERS = 1000
N = 6
n_dim = 3 # State dimension
pos_init = (np.random.rand(N, n_dim) - 0.5)*5
pos_init[:, 2] = 0.
comm_time = 1/30 # Comunication time
euler_step = 0.005 # Integration step
L = 2


if N == 4: # Square
    D = np.sqrt(2)*L
    distances = [
                [0, L, D, L],
                [L, 0, L, D],
                [D, L, 0, L],
                [L, D, L, 0]
                ]
    
if N == 5: # Pentagon
    D = L*(1+np.sqrt(5))/2
    distances = [[0, L, D, D, L],
                [L, 0, L, D, D],
                [D, L, 0, L, D],
                [D, D, L, 0, L],
                [L, D, D, L, 0]]
    
if N == 6: #Hexagon
    D = 2*L
    H = np.sqrt(3)*L
    distances = [[0, L , 0, D, H, L],
                [L, 0, L, 0, D, 0],
                [0, L, 0, L, 0, D],
                [D, 0, L, 0, L, 0],
                [H, D, 0, L, 0, L],
                [L, 0, D, 0, L, 0]]
    
############ Moving leader part ###############
# Moving leaders
agent_types = np.zeros((N)) # Followers -> 0
for i in range(N):
    if i%2 == 1:
        agent_types[i] = 1 # Leaders -> 1


u_lin = np.zeros((n_dim))
u_lin[0] = 3
u_lin[1] = 0
u_lin[2] = 0


def generate_launch_description():
    launch_description = [] # Append here your nodes

    # RVIZ Node

    # initialize launch description with rviz executable
    rviz_config_dir = get_package_share_directory('formation_control')
    rviz_config_file = os.path.join(rviz_config_dir, 'rviz_config.rviz')

    launch_description.append(
        Node(
            package='rviz2',
            executable='rviz2', 
            arguments=['-d', rviz_config_file],
            # output='screen',
            # prefix='xterm -title "rviz2" -hold -e'
        )
    )

    for i in range(N):
        # Agents
        launch_description.append(
            Node(
                package='formation_control',
                namespace =f'agent_{i}',
                executable='the_agent',
                parameters=[{ # dictionary
                                'agent_id': i, 
                                'pos_init': pos_init[i].tolist(),
                                'distances': distances[i], 
                                'max_iters': MAXITERS,
                                'comm_time': comm_time,
                                'euler_step': euler_step,
                                'type' : agent_types[i].tolist(),
                                'u_lin' : u_lin.tolist(),
                                }],
                output='screen',
                prefix=f'xterm -title "agent_{i}" -hold -e',
            ))

    for i in range(N):
        # Launch visualizer
        launch_description.append(
            Node(
                package='formation_control',
                namespace=f'agent_{i}',
                executable='visualizer',
                parameters=[{
                    'agent_id': i,
                    'comm_time': comm_time,
                    'type' : agent_types[i].tolist(),
                    }]
                )
            )
        
    # Plotter node
    flattened_distances = [item for sublist in distances for item in sublist]
    launch_description.append(
        Node(
            package='formation_control',
            executable='the_plotter',
            parameters=[{ # dictionary
                            'max_iters': MAXITERS,
                            'distance_matrix': flattened_distances,
                            'comm_time': comm_time
                            }],
            output='screen',
            prefix=f'xterm -title "the_plotter" -hold -e',
        )
    )

    return LaunchDescription(launch_description)