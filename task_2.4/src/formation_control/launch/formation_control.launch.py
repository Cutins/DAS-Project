from launch import LaunchDescription
from launch_ros.actions import Node
import numpy as np
import networkx as nx
import os
from ament_index_python.packages import get_package_share_directory

MAXITERS = 5000
N = 6
n_dim = 3 # State dimension
pos_init = (np.random.rand(N, 3) - 0.5) *0.1
pos_init[:, 2] = 0.
comm_time = 1/30 # Comunication time
euler_step = 0.005 # Integration step
L = 2

# Moving leaders
agent_types = np.zeros((N)) # Followers -> 0
for i in range(N):
    if i%2 == 1:
        agent_types[i] = 1 # Leaders -> 1


# n_leaders = int(N/2) + (N%2)
# agent_types[n_follower:] = 1 # Leaders -> 1

# input = np.zeros((n_dim))
# input[0] = 5
# input[1] = 0
# input[2] = 0
move = 3   # Raggio del cercio

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
    

######### The obstacle #########
N_obstacles = 8
pos_obs = np.zeros((N_obstacles, n_dim))
for i in range(N_obstacles):
    if i%2 == 0: #even
        pos_obs[i] = [2+(i/2), 4, 0]
    else:   #odd
        pos_obs[i] = [-2-(i/2), 4, 0]

# pos_obs[0] = [2, 5, 0]
# pos_obs[1] = [-2, 5, 0]



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


    # for i_foll in range(n_follower):
    #     launch_description.append(
    #         Node(
    #             package='formation_control',
    #             namespace =f'agent_{i_foll}',
    #             executable='the_agent',
    #             parameters=[{ # dictionary
    #                             'agent_id': i_foll, 
    #                             'pos_init': pos_init[i_foll].tolist(),
    #                             'distances': distances[i_foll], 
    #                             'max_iters': MAXITERS,
    #                             'comm_time': comm_time,
    #                             'euler_step': euler_step,
    #                             'type' : 0, # Follower
    #                             'input' : input#.tolist(),
    #                             }],
    #             output='screen',
    #             prefix=f'xterm -title "agent_{i_foll}" -hold -e',
    #         ))
        
    # for i in range(n_leader):
    #     i_lead = n_follower+i
    #     launch_description.append(
    #         Node(
    #             package='formation_control',
    #             namespace =f'agent_{i_lead}',
    #             executable='the_agent',
    #             parameters=[{ # dictionary
    #                             'agent_id': i_lead, 
    #                             'pos_init': pos_init[i_lead].tolist(),
    #                             'distances': distances[i_lead], 
    #                             'max_iters': MAXITERS,
    #                             'comm_time': comm_time,
    #                             'euler_step': euler_step,
    #                             'type' : 1, # Leader
    #                             'input' : input#.tolist(),
    #                             }],
    #             output='screen',
    #             prefix=f'xterm -title "agent_{i_lead}" -hold -e',
    #         ))
    
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
                                'move' : move,#.tolist(),
                                'N_obstacles' : N_obstacles,
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
                    'agent_types' : agent_types.tolist(),
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

    # The obstacles
    for i in range(N_obstacles):
        #Obstacles
        launch_description.append(
            Node(
                package='formation_control',
                namespace =f'obstacle_{i}',
                executable='the_obstacle',
                parameters=[{ # dictionary
                                'obstacle_id': i, 
                                'pos_init': pos_obs[i].tolist(),
                                'comm_time': comm_time,
                                }],
                # output='screen',
                # prefix=f'xterm -title "obstacle_{i}" -hold -e',
            )
        )

        # Launch visualizer for obstacles
        launch_description.append(
            Node(
                package='formation_control',
                namespace=f'obstacle_{i}',
                executable='visualizer_for_obstacle',
                parameters=[{
                    'agent_id':i,
                    'comm_time':comm_time,
                    }]
                )
            )

    return LaunchDescription(launch_description)