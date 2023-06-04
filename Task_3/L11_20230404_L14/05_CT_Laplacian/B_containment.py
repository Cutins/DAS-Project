#
# Containment Algorithm
# Lorenzo Pichierri, Ivano Notarnicola
# Bologna, 03/04/2023
#
import numpy as np
import matplotlib.pyplot as plt
import control
import networkx as nx
from functions import containment as animation

np.random.seed(10)

# The leaders will move according to a sinusoidal function
def waves(amp, omega, phi, t, n_x, n_leaders):
	u = [amp*np.sin(omega*t+phi) for _ in range(n_x*n_leaders)]
	return np.array(u)

ANIMATION = True

Tmax = 10.0 	# simulation time
NN = 15 		# number of agents
n_x = 2 		# dimension of x_i 
n_leaders = 5

p_ER = 0.7		# edge probability

I_NN = np.eye(NN)
I_nx = np.eye(n_x)
I_NN_nx = np.eye(n_x*NN)
O_NN = np.ones((NN,1))
	
# ER Network generation
while 1:
	graph_ER = nx.binomial_graph(NN,p_ER)
	Adj = nx.adjacency_matrix(graph_ER).toarray()
	
	test = np.linalg.matrix_power((I_NN+Adj),NN)
	
	if np.all(test>0):
		print("the graph is connected\n")
		break
	else:
		print("the graph is NOT connected\n")


DEGREE = np.sum(Adj,axis=0) 
D_IN = np.diag(DEGREE)
L_IN = D_IN - Adj.T

L_f = L_IN[0:NN-n_leaders,0:NN-n_leaders] # To take the upper-left matrix
L_fl = L_IN[0:NN-n_leaders,NN-n_leaders:] # To take the upper-right matrix

# leaders dynamics
LL = np.concatenate((L_f, L_fl), axis = 1)
LL = np.concatenate((LL, np.zeros((n_leaders,NN))), axis = 0)

# replicate for each dimension
LL_kron = np.kron(LL,I_nx)

x_init = np.vstack((np.ones((n_x*n_leaders,1)),np.zeros((n_x*(NN-n_leaders),1))))
x_init += 5*np.random.rand(n_x*NN,1)

BB_kron = np.zeros((NN*n_x,n_leaders*n_x))
BB_kron[(NN-n_leaders)*n_x:,:] = np.identity(n_x*n_leaders, dtype=int)

A = -LL_kron
B = BB_kron
C = np.identity(np.size(LL_kron,axis = 0)) # to comply with StateSpace syntax

################
# Followers integral Action

k_i = 10 #20
k_i = 0
K_I = - k_i*I_NN_nx

LL_ext_up = np.concatenate((LL_kron, K_I), axis = 1)
LL_ext_low = np.concatenate((LL_kron, np.zeros(LL_kron.shape)), axis = 1)
LL_ext = np.concatenate((LL_ext_up, LL_ext_low), axis = 0)

# include integral state
x_init = np.concatenate((x_init,np.zeros((n_x*NN,1))))
BB_kron = np.concatenate((BB_kron, np.zeros((NN*n_x,n_leaders*n_x))), axis = 0)

A = -LL_ext
B = BB_kron
C = np.identity(np.size(LL_ext,axis = 0)) # to comply with StateSpace syntax

################

sys = control.StateSpace(A,B,C,0) # dx = -L^IN x + B u

dt = 0.01
horizon = np.arange(0.0, Tmax, dt)

# Leaders input
(amp, omega, phi) = (6, 2, 0)
u = waves(amp, omega, phi, horizon, n_x, n_leaders)
# u = 10*np.ones((n_x*n_leaders, len(horizon)))
# u = np.zeros((n_x*n_leaders, len(horizon)))

(T, yout, xout) = control.forced_response(sys, X0=x_init, U=u, T=horizon, return_x=True)

################################################
# Drawings

plt.figure(1)
label = []
for ii in range(0,n_x*NN,2):
	if ii<n_x*(NN-n_leaders):
		color = 'tab:blue'	# followers
	else:
		color = 'tab:red' 	# leaders
		
	plt.plot(horizon, xout[ii], color=color)
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

	plt.plot(horizon, xout[ii], color=color)
	label.append(f'$x_{int(ii/2)}$')

plt.legend(label)

plt.title("Evolution of the local estimates y-axis")
plt.xlabel("$t$")
plt.ylabel("$x_i^t$")

if ANIMATION: # animation (0 to avoid animation)
	if n_x == 2: 
		plt.figure(3)
		animation(xout,NN,n_x,n_leaders, horizon = T, dt=10)

plt.show()
