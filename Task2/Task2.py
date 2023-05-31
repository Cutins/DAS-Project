# from launch import LaunchDescription
# from launch_ros.actions import Node
import numpy as np
import matplotlib.pyplot as plt
# from functions import formation as animation

np.random.seed(10)

# ANIMATION = True
MAXITERS = 10
n_x = 3
dt = 0.01 # integration time = communication time
N_AGENTS = 5

#### Select formation shape
Formation = "pentagon" # {"square" "pentagon" "hexagon"}

#### Formations NxN distance matrices
distances = np.zeros((N_AGENTS, N_AGENTS, N_AGENTS))
L = 2

# Square
if Formation == "square": 
    D = np.sqrt(2)*L
    distances_2d = [[0, L, D, L],
                    [L, 0, L, D],
                    [D, L, 0, L],
                    [L, D, L, 0]]
    
    position = np.array([[0, 0],
                [L, 0],
                [L, L],
                [0, L],
                [0,0]])

    distances_2d = np.asarray(distances_2d)
    distances = np.asarray(distances_2d)

# Pentagon
if Formation == "pentagon": 
    D = L*(1+np.sqrt(5))/2
    distances_2d = [[0, L, D, D, L],
                    [L, 0, L, D, D],
                    [D, L, 0, L, D],
                    [D, D, L, 0, L],
                    [L, D, D, L, 0]]
    
    position = np.array([[0,0],
                [L,0],
                [L+L*np.cos(np.pi/10), L*np.sin(np.pi/10)],
                [L/2, L*np.sin(np.pi/5)],
                [-L*np.cos(np.pi/10), L*np.sin(np.pi/10)],
                [0,0]])
        
    distances_2d = np.asarray(distances_2d)
    distances = np.asarray(distances_2d)

# Hexagon
if Formation == "hexagon":
    D = 2*L
    H = np.sqrt(3)*L
    distances_2d = [[0, L , 0, D, H, L],
                    [L, 0, L, 0, D, 0],
                    [0, L, 0, L, 0, D],
                    [D, 0, L, 0, L, 0],
                    [H, D, 0, L, 0, L],
                    [L, 0, D, 0, L, 0]] 
    
    position = np.array([[0, 0],
                [L, 0],
                [L+L/2, L*np.sqrt(3)/2],
                [L, L*np.sqrt(3)],
                [0, L*np.sqrt(3)],
                [-L/2, L*np.sqrt(3)/2],
                [0,0]])
    
    distances_2d = np.asarray(distances_2d)
    distances = np.asarray(distances_2d)

# Adjacency matrix
Adj = distances_2d > 0

# definite initial positions
x_init = 10*np.random.rand(n_x*N_AGENTS)

launch_description = [] # Append nodes

for agent in range(N_AGENTS):
    neighs = np.nonzero(Adj[:, agent])[0].tolist()
    print(f'Neighbours of {agent} are: {neighs}\n')

def plot_position(position):
    plt.plot(position[:,0],position[:,1], marker='o', color='r', markersize=20, fillstyle='none')
    plt.show()

plot_position(position)

