
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import networkx as nx
np.random.seed(0)

# Quadratic Function
def quadratic_fn(x,Q,r):
    fval = 0.5*(x.T@Q@x) + r.T@x
    fgrad = Q@x + r

    return fval, fgrad

# graph_type = {"Cycle", "Path", "Star"}
graph_type = "Cycle"

#####################################################################################
# Usefull constants
MAXITERS = 3*int(1e3)
NN = 5
dd = 3

#####################################################################################
#  Generate Network Graph

if graph_type == "Cycle":
	G = nx.path_graph(NN)
	G.add_edge(NN-1,0)
	nx.draw_circular(G, with_labels=True)


if graph_type == "Path":
	G = nx.path_graph(NN)
	nx.draw(G)
	
if graph_type == "Star":
	G = nx.star_graph(NN-1)
	nx.draw(G, with_labels=True)

plt.show()

I_NN = np.identity(NN, dtype=int)

while 1:
	Adj = nx.adjacency_matrix(G)
	Adj =Adj.toarray()	

	test = np.linalg.matrix_power((I_NN+Adj),NN)
	
	if np.all(test>0):
		print("the graph is connected\n")
		break 
	else:
		print("the graph is NOT connected\n")
		quit()

#####################################################################################
# Compute mixing matrices
# Metropolis-Hastings method

degree = np.sum(Adj, axis=0)
W = np.zeros((NN,NN))

for ii in range(NN):
   Nii = np.nonzero(Adj[ii])[0]
   
   for jj in Nii:
      W[ii,jj] = 1/(1+np.max([degree[ii],degree[jj]]))

   W[ii,ii] = 1-np.sum(W[ii,:])

# print(W)

print('Row Stochasticity {}'.format(np.sum(W,axis=1)))
print('Col Stochasticity {}'.format(np.sum(W,axis=0)))

#####################################################################################
# Declare Cost Variables
if 0: # We are not sure Q to positive definite
	Q = 10*(np.random.rand(NN,dd,dd)) # np.zeros((NN,dd,dd))
	for ii in range(NN):
		Q[ii] = 0.5*(Q[ii].T*Q[ii])

if 1:
	Q = np.zeros((NN,dd,dd)) # positive definite
	for ii in range(NN):
		T = scipy.linalg.orth(np.random.rand(dd,dd))
		D = np.diag(np.random.rand(dd))*10
		Q[ii] = T.T@D@T
		# print(np.linalg.eigvals(Q[ii]))

R = 10*(np.random.rand(NN,dd)-1) # np.zeros((NN,dd)) 

# Compute the optimal solution
Q_centr = np.sum(Q,axis=0) #shape (d,d)
R_centr = np.sum(R,axis=0) #shape (d,)

# Ottimo dalle condizioni iniziali
xopt = -np.linalg.inv(Q_centr)@R_centr #-R/D #shape (d,)
fopt = 0.5*xopt@Q_centr@xopt+R_centr@xopt #shape ()
print('The optimal cost is: {:.4f}'.format(fopt))

# Declare algorithmic variables
XX = np.zeros((NN,MAXITERS,dd))
SS = np.zeros((NN,MAXITERS,dd))

XX_init = 10*np.random.rand(NN,dd)
XX[:,0,:] = XX_init # print('Shape:\n {}'.format(XX[:,0].shape))
FF = np.zeros((MAXITERS))


# Initialization of SS for the GT algorithm
# s_i^0 = \nabla f_i(x_i^0)
for ii in range(NN):
	_, SS[ii,0,:] = quadratic_fn(XX[ii,0,:], Q[ii,:,:], R[ii,:])

# quit(0)

###############################################################################
# GO!
stepsize0 = 1e-3

for kk in range (MAXITERS-1):
	# stepsize = stepsize0/(kk+1) # Diminishing stepsize
	stepsize = stepsize0 # Constant stepsize

	if (kk % 10) == 0:
		print("Iteration {:3d}".format(kk), end="\n")
	
	for ii in range (NN):
		Nii = np.nonzero(Adj[ii])[0]

		f_ii, grad_f_ii = quadratic_fn(XX[ii,kk,:], Q[ii,:,:], R[ii,:])

		XX[ii,kk+1,:] = W[ii,ii]*XX[ii,kk,:] - stepsize*SS[ii,kk,:]
		for jj in Nii:
			XX[ii,kk+1,:] += W[ii,jj]*XX[jj,kk,:]

		_, grad_f_ii_p = quadratic_fn(XX[ii,kk+1,:], Q[ii,:,:], R[ii,:])

		SS[ii,kk+1,:] = W[ii,ii]*SS[ii,kk,:] +(grad_f_ii_p-grad_f_ii)
		for jj in Nii:
			SS[ii,kk+1,:] += W[ii,jj]*SS[ii,kk,:]

		FF[kk] += f_ii

# Terminal iteration
for ii in range(NN):
	f_ii, _ = quadratic_fn(XX[ii,-1,:], Q[ii,:,:], R[ii,:])
	FF[-1] += f_ii 

###############################################################################
# generate N random colors
colors = {}
for ii in range(NN):
	colors[ii] = np.random.rand(3)

###############################################################################
# Figure 1 : Evolution of the local estimates
if 1:
	plt.figure()
	plt.plot(np.arange(MAXITERS), np.tile(xopt,(MAXITERS,1)), ':', linewidth=3)

	for ii in range(NN):
		plt.plot(np.arange(MAXITERS), XX[ii,:,:], color=colors[ii])
		
	plt.xlabel(r"iterations $k$")
	plt.ylabel(r"$x^\star$, $\bar{x}^k$, $x_i^k$")
	plt.title("Evolution of the local estimates")
	plt.grid()

###############################################################################
# Figure 2 : Cost Evolution
if 1: 
	plt.figure()
	plt.plot(np.arange(MAXITERS), np.repeat(fopt,MAXITERS), '--', linewidth=3, label='Optimal cost function')
	plt.plot(np.arange(MAXITERS), FF, label='Evolution of cost function')
	plt.xlabel(r"iterations $k$")
	plt.ylabel(r"$\sum_{i=1}^N f_i(x_i^k)$, $f^\star$")
	plt.title("Evolution of the cost")
	plt.legend()
	plt.grid()


###############################################################################
# Figure 3 : Cost Error Evolution
if 1:
	plt.figure()
	plt.semilogy(np.arange(MAXITERS), np.abs(FF-np.repeat(fopt,MAXITERS)), '--', linewidth=3)
	plt.xlabel(r"iterations $k$")
	plt.ylabel(r"$|\sum_{i=1}^N f_i(x_i^k) - f^\star|$")
	plt.title("Evolution of the cost error")
	plt.grid()

plt.show()

