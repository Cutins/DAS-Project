from launch import LaunchDescription
from launch_ros.actions import Node
import numpy as np
import networkx as nx
import os
from ament_index_python.packages import get_package_share_directory

MAXITERS = 1000
N = 6
n_dim = 3 # State dimension
pos_init = (np.random.rand(N, n_dim) - 0.5)
pos_init[:, 2] = 0.
comm_time = 1/30 # Comunication time
euler_step = 0.001 # Integration step
L = 2



################################# FORMATION ########################################
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
    
    
################################ CONTAINMENT DYNAMICS ########################################

adj = np.array(distances) > 0
degree_out = np.sum(adj, axis=-1)
D_out = np.eye(N) * degree_out
n_follower = int(N/2)
n_leader = N - n_follower

agent_types = np.zeros((N,1))
agent_types[n_leader:] = 1

L_out = D_out - adj
L_f     = L_out[:n_follower, :n_follower]
L_fl    = L_out[:n_follower, n_follower:]

# Laplacian dynamics
LL = np.zeros((N, N))
LL[:n_follower, :n_follower] = L_f
LL[:n_follower, n_follower:] = L_fl

# replicate for each dimension -> kronecker product
LL_kron = np.kron(LL, np.eye(n_dim)) # Shape: 3N x 3N

## followers integral Action
k_integral = 0

LL_kron_dim = LL_kron.shape[0] # [n_dim * N]
LL_PI = np.zeros((2*LL_kron_dim, 2*LL_kron_dim))
LL_PI[:LL_kron_dim, :LL_kron_dim] = LL_kron
LL_PI[:LL_kron_dim, LL_kron_dim:] = LL_kron * k_integral
LL_PI[LL_kron_dim:, :LL_kron_dim] = 0
LL_PI[LL_kron_dim:, LL_kron_dim:] = - np.eye(LL_kron_dim)
LL_PI = -LL_PI
print(LL_PI[(n_dim*0):(n_dim*(0+1))].flatten().tolist())
################################ LEADER DYNAMICS ########################################
# Initialize a linear input along x
proportionial_u = np.zeros((n_dim))
proportionial_u[0] = 1

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


    for i_fol in range(n_follower):

        launch_description.append(
            Node(
                package='formation_control',
                namespace =f'agent_{i_fol}',
                executable='the_agent',
                parameters=[{ # dictionary
                                'agent_id': i_fol, 
                                'pos_init': pos_init[i_fol].tolist(),
                                'distances': distances[i_fol], 
                                'max_iters': MAXITERS,
                                'comm_time': comm_time,
                                'euler_step': euler_step,
                                'type': 0, # 0 -> follower, 1 -> leader
                                'laplacian_PI' : LL_PI[(n_dim*i_fol):(n_dim*(i_fol+1))].flatten().tolist(),
                                'linear_u' : np.zeros_like(proportionial_u).tolist(),
                                }],
                output='screen',
                prefix=f'xterm -title "agent_{i_fol}" -hold -e',
            ))
        
    for i in range(n_leader):
        i_lead = n_follower+i
        launch_description.append(
            Node(
                package='formation_control',
                namespace =f'agent_{i_lead}',
                executable='the_agent',
                parameters=[{ # dictionary
                                'agent_id': i_lead, 
                                'pos_init': pos_init[i_lead].tolist(),
                                'distances': distances[i_lead], 
                                'max_iters': MAXITERS,
                                'comm_time': comm_time,
                                'euler_step': euler_step,
                                'type': 1, # 0 -> follower, 1 -> leader
                                'laplacian_PI' : [0],
                                'linear_u' : proportionial_u.tolist(),
                                }],
                output='screen',
                prefix=f'xterm -title "agent_{i_lead}" -hold -e',
            ))

    # for i in range(N):

    #     launch_description.append(
    #         Node(
    #             package='formation_control',
    #             namespace =f'agent_{i}',
    #             executable='the_agent',
    #             parameters=[{ # dictionary
    #                             'agent_id': i, 
    #                             'pos_init': pos_init[i].tolist(),
    #                             'distances': distances[i], 
    #                             'max_iters': MAXITERS,
    #                             'comm_time': comm_time,
    #                             'euler_step': euler_step,
    #                             'type': agent_types[i],
    #                             'laplacian_PI' : LL_PI[(n_dim*i):(n_dim*(i+1))].tolist()
    #                             }],
    #             output='screen',
    #             prefix=f'xterm -title "agent_{i}" -hold -e',
    #         ))
    for i in range(N):
        # Launch visualizer
        launch_description.append(
            Node(
                package='formation_control',
                namespace=f'agent_{i}',
                executable='visualizer',
                parameters=[{
                    'agent_id':i,
                    'comm_time':comm_time,
                    # 'n_leaders':n_leaders,
                    }]
                )
            )
        

    flattened_distances = [item for sublist in distances for item in sublist]
    launch_description.append(
        Node(
            package='formation_control',
            executable='the_plotter',
            parameters=[{ # dictionary
                            'max_iters': MAXITERS,
                            # 'n_agents': N,
                            'distance_matrix': flattened_distances,
                            'comm_time': comm_time
                            }],
            output='screen',
            prefix=f'xterm -title "the_plotter" -hold -e',
        )
    )

    return LaunchDescription(launch_description)