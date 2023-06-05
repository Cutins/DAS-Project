#
# Laplacian Consensus Algorithm
# Ivano Notarnicola, Lorenzo Pichierri
# Bologna, 03/04/2023
#
import numpy as np
import matplotlib.pyplot as plt
import control	# To simulate dynamical system in C-T
import networkx as nx

Tmax = 10.0	# simulation time
NN = 10 	# number of agents
n_x = 2 	# dimension of x_i 

p_ER = 0.5 	# edge probability

I_NN = np.eye(NN)
I_nx = np.eye(n_x)
I_NN_nx = np.eye(n_x*NN)
O_NN = np.ones((NN,1))

# ER Network generation
while 1:
	graph_ER = nx.binomial_graph(NN,p_ER)
	Adj = nx.adjacency_matrix(graph_ER).toarray()
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
# To check if it's weighted balanced
DEGREE_OUT = np.sum(Adj,axis=1)
print(f'D_in = {DEGREE}')
print(f'D_out = {DEGREE_OUT}')

x_init = np.random.rand(n_x*NN,1)
# x_init = 10*(np.random.rand(NN,n_x)-0.5)

# To compute the average...
x_init_ag = np.reshape(x_init,(NN,n_x))
x0_mean = np.mean(x_init_ag,axis=0)
x0_mean = np.array([x0_mean])

A = -np.kron(L_IN,I_nx)	#Product [10x10]*[3x3]
B = np.ones((NN*n_x,1), dtype=int) # to comply with StateSpace syntax
C = I_NN_nx # to comply with StateSpace syntax
sys_consensus = control.StateSpace(A,B,C,0) # dx = -L^IN x

dt = 0.1
horizon = np.arange(0.0, Tmax, dt)

(T, yout, xout) = control.initial_response(sys_consensus,X0=x_init,T=horizon, return_x=True)
print(xout)
print(np.shape(xout))

################################################
# Drawings

# Generate Figure
plt.figure(1)
for x in xout:
	plt.plot(T, x)

# Plot mean values
plt.plot(T, np.repeat(x0_mean, len(T), axis = 0),  '--', linewidth=3)

plt.title("Evolution of the local estimates")
plt.xlabel("$t$")
plt.ylabel("$x_i^t$")


plt.figure(2)
for i in range(0, np.shape(xout)[0]-1, 2):
	plt.plot(xout[i], xout[i+1])

plt.plot(x0_mean.T[0],x0_mean.T[1], 'o', linewidth=20)


plt.show()
