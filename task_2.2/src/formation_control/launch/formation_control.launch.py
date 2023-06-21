from launch import LaunchDescription
from launch_ros.actions import Node
import numpy as np
import networkx as nx
import os
from ament_index_python.packages import get_package_share_directory

MAXITERS = 1200
N = 6
n_dim = 3 # State dimension

# Uncomment for set initial position in a row, better to simulate the collision
# pos_init = np.zeros((N, 3))
# pos_init[:, 0] = [((idx)/N) for idx in range(N)]
# pos_init[:, 1] = [0.01*((idx)/N) for idx in range(N)]
pos_init = (np.random.rand(N, 3) - 0.5) *0.1

pos_init[:, 2] = 0.
comm_time = 1/30 # Comunication time
euler_step = 0.001 # Integration step
L = 2

_3d_formation = False

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
    
if _3d_formation == True: # Square-based pyramid
    N = 5
    pos_init = (np.random.rand(N, 3) - 0.5) *0.1
    L = 3
    H = 4
    D1 = np.sqrt(2)*L
    D = np.sqrt(H**2+(L/2)**2)
    distances = [[0, L, D1, L, D],
                 [L, 0, L, D1, D],
                 [D1, L, 0, L, D],
                 [L, D1, L, 0, D],
                 [D, D, D, D, 0]]


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
                                }],
                output='screen',
                prefix=f'xterm -title "agent_{i}" -hold -e',
            ))
        
        # Launch visualizer
        launch_description.append(
            Node(
                package='formation_control',
                namespace=f'agent_{i}',
                executable='visualizer',
                parameters=[{
                    'agent_id':i,
                    'comm_time':comm_time,
                    }]
                )
            )
        
    # Plotter Node
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