#
# Containment Algorithm DT
# Lorenzo Pichierri, Ivano Notarnicola
# Bologna, 03/04/2023
#
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from functions import waves, animation

from pprint import pprint

np.random.seed(5)

# Bool vars
ANIMATION = True

#################################################################

TT = 10.0	# Simulation time
NN = 15 	# number of agents
n_x = 2 	# dimension of x_i 
n_leaders = 5

p_ER = 0.7

I_NN = np.eye(NN)
I_nx = np.eye(n_x)
I_NN_nx = np.eye(n_x*NN)
O_NN = np.ones((NN,1))

# Generate a Connected graph
while 1:
	G = nx.binomial_graph(NN,p_ER)
	Adj = nx.adjacency_matrix(G).toarray()
  
	# test connectivity
	test = np.linalg.matrix_power((I_NN+Adj),NN)
	
	if np.all(test>0):
		print("the graph is connected\n")
		break
	else:
		print("the graph is NOT connected\n")


DEGREE = np.sum(Adj,axis=0) 
D_IN = np.diag(DEGREE)
L_IN = D_IN - Adj.T

L_f = L_IN[0:NN-n_leaders,0:NN-n_leaders]
L_fl = L_IN[0:NN-n_leaders,NN-n_leaders:]

# followers dynamics
LL = np.concatenate((L_f, L_fl), axis = 1)

# leaders dynamics
LL = np.concatenate((LL, np.zeros((n_leaders,NN))), axis = 0)

# replicate for each dimension -> kronecker product
LL_kron = np.kron(LL,I_nx)

# Initiate the agents' state
XX_init = np.vstack((np.ones((n_x*n_leaders,1)),np.zeros((n_x*(NN-n_leaders),1))))
XX_init += 10*np.random.rand(n_x*NN,1)

# Consider only the leaders in the B matrix
BB_kron = np.zeros((NN*n_x,n_leaders*n_x))
BB_kron[(NN-n_leaders)*n_x:,:] = np.identity(n_x*n_leaders, dtype=int)

################################################
## followers integral Action

k_i = 0 #10
K_I = - k_i*I_NN_nx

# Setup the extended dynamics
LL_ext_up = np.concatenate((LL_kron, K_I), axis = 1)
LL_ext_low = np.concatenate((LL_kron, np.zeros(LL_kron.shape)), axis = 1)
LL_ext = np.concatenate((LL_ext_up, LL_ext_low), axis = 0)

# extende the initial state with the integral state
XX_init = np.concatenate((XX_init,np.zeros((n_x*NN,1))))
BB_kron = np.concatenate((BB_kron, np.zeros((NN*n_x,n_leaders*n_x))), axis = 0)

A = -LL_ext
B = BB_kron
print(BB_kron.shape)

print(f'Autovalori di A\n{np.linalg.eigvals(A)}')
pprint(f'Matrice A\n{A}')
pprint(f'Matrice B\n{B}')
################################################
# CONTAINMENT Dynamics

dt = 0.005 	# Sampling time
horizon = np.arange(0.0, TT, dt)

XX = np.zeros((A.shape[1],len(horizon)))
XX[:,0] = XX_init.T

# Leaders input: null, sinusoidal
(amp, omega, phi) = (6, 2, 0)
# UU = waves(amp, omega, phi, horizon, n_x, n_leaders)
UU = np.zeros((n_leaders*n_x, len(horizon)))

for tt in range(len(horizon)-1):
	XX[:, tt + 1] = XX[:, tt] + dt*(A @ XX[:, tt] + B @ UU[:, tt])


################################################
# Drawings

plt.figure(1)
label = []
for ii in range(0,n_x*NN,2):
	if ii<n_x*(NN-n_leaders):
		color = 'tab:blue'	# followers
	else:
		color = 'tab:red' 	# leaders
		
	plt.plot(horizon, XX[ii,:], color=color)
	label.append(f'$x_{int(ii/2)}$')

plt.legend(label)

plt.title("Evolution of the local estimates x-axis")
plt.xlabel("$t$")
plt.ylabel("$x_i^t$")

plt.figure(2)
label = []
for ii in range(1,n_x*NN,2):
	if ii<n_x*(NN-n_leaders):
		color = 'tab:blue'	# followers
	else:
		color = 'tab:red'	# leaders

	plt.plot(horizon, XX[ii,:], color=color)
	label.append(f'$x_{int(ii/2)}$')

plt.legend(label)

plt.title("Evolution of the local estimates y-axis")
plt.xlabel("$t$")
plt.ylabel("$x_i^t$")

if ANIMATION: 
  if n_x == 2: 
    plt.figure(3)
    animation(XX, NN, n_x, n_leaders, horizon, dt = 50)
  else:
    print('Animation allowed only for bi-dimensional agent')

plt.show()
