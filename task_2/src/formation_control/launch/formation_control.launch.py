from launch import LaunchDescription
from launch_ros.actions import Node
import numpy as np
import networkx as nx

MAXITERS = 50
N = 5
pos_init = (np.random.rand(N, 3) - 0.5)
pos_init[:, 2] = 0.
dt = 1e-2
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

def generate_launch_description():
    launch_description = [] # Append here your nodes

    for i in range(N):

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
                                'integration_step': dt
                                }],
                output='screen',
                prefix=f'xterm -title "agent_{i}" -hold -e',
            ))

    return LaunchDescription(launch_description)