'''
Task 1_1 Funzionante

'''


import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
import time
import networkx as nx
import scipy

MAXITERS = int(1e2)     # Number of iterations
N_AGENTS = 5            # Number of agents
D_AGENTS = 3            # Dimensionality of agents
STEPSIZE = 1e-2

#####################################################################################
#  Function
def quad_funct(X, QQ, R):
    """
        It's works with tensor ora matrix. Also with N_AGENTS as dimension
		X -> (D,)   QQ -> (D,D)  R-> (D,)
        fval = 0.5*QQ*(X^2) + R*X
        fgrad = QQ*X + R
    """ 
    fval = 0.5*(X.T@QQ@X) + R.T@X
    fgrad = QQ@X + R
    return fval, fgrad
    '''
    if len(X.shape) == 1: 
        fval = 0.5*(X[..., None].T@QQ@X) + R[..., None].T@X	# (,D)(D,D)(D,) + (,D)(D,)
        fgrad = QQ@X + R
    elif len(X.shape) == 2:
	# X -> (N,D)   QQ -> (N,D,D)  R-> (N,D)
        R_T = np.transpose(R, axes=(0, 2, 1))
        X_T = X.T
        fval = 0.5*(X_T@QQ@X) + R.T@X
        fgrad = QQ@X + R
    else:
        print(f'Dimension of X is: {len(X.shape)}')
        raise 'Wrong dimensions!'
        
    return fval, fgrad
    '''

#####################################################################################
#  Generate Network Graph
GRAPH_TYPE = "Cycle"    # {"Cycle", "Path", "Star"}

if GRAPH_TYPE == "Cycle":
	G = nx.path_graph(N_AGENTS)
	G.add_edge(N_AGENTS-1,0)
	nx.draw_circular(G, with_labels=True)

if GRAPH_TYPE == "Path":
	G = nx.path_graph(N_AGENTS)
	nx.draw(G)
	
if GRAPH_TYPE == "Star":
	G = nx.star_graph(N_AGENTS-1)
	nx.draw(G, with_labels=True)

plt.show()

ID_AGENTS = np.identity(N_AGENTS, dtype=int)

while 1:
	ADJ = nx.adjacency_matrix(G)
	ADJ = ADJ.toarray()	

	test = np.linalg.matrix_power((ID_AGENTS+ADJ),N_AGENTS)
	
	if np.all(test>0):
		print("the graph is connected\n")
		break 
	else:
		print("the graph is NOT connected\n")
		quit()


#####################################################################################
# METROPOLIS HASTING
degree = np.sum(ADJ, axis=0)
WW = np.zeros((N_AGENTS,N_AGENTS))

for ii in range(N_AGENTS):
   Nii = np.nonzero(ADJ[ii])[0]
   
   for jj in Nii:
      WW[ii,jj] = 1/(1+np.max([degree[ii],degree[jj]]))

   WW[ii,ii] = 1-np.sum(WW[ii,:])

print(WW)

print('Row Stochasticity {}'.format(np.sum(WW,axis=1)))
print('Col Stochasticity {}'.format(np.sum(WW,axis=0)))

#####################################################################################

X = np.zeros((MAXITERS, N_AGENTS, D_AGENTS))
S = np.zeros((MAXITERS, N_AGENTS, D_AGENTS))
Cost = np.zeros((MAXITERS, N_AGENTS))
Grad = np.zeros((MAXITERS, N_AGENTS, D_AGENTS))

# Declare Cost Variables
QQ = np.zeros((N_AGENTS, D_AGENTS, D_AGENTS))
R = 10*(np.random.rand(N_AGENTS, D_AGENTS)) # np.zeros((NN,dd)) 

if 0: # We are not sure Q to positive definite
	QQ = 10*(np.random.rand(N_AGENTS, D_AGENTS, D_AGENTS)+1) # np.zeros((N_AGENTS,D_AGENTS,D_AGENTS))
	for ii in range(N_AGENTS):
		QQ[ii] = 0.5*(QQ[ii].T*QQ[ii])

if 1:
	QQ = np.zeros((N_AGENTS, D_AGENTS, D_AGENTS)) # positive definite
	for ii in range(N_AGENTS):
		T = scipy.linalg.orth(np.random.rand(D_AGENTS, D_AGENTS))
		D = np.diag(np.random.rand(D_AGENTS))*10
		QQ[ii] = T.T@D@T
		# print(np.linalg.eigvals(Q[ii]))

X[0] = 10*np.random.rand(N_AGENTS, D_AGENTS)
for ii in range(N_AGENTS):
	# s_i^0 = \nabla f_i(x_i^0)
	_, S[0, ii] = quad_funct(X[0, ii], QQ[ii], R[ii])
	
#####################################################################################
# Compute the optimal solution for plots
QQ_centr = np.sum(QQ,axis=0)        # shape (D_AGENTS, D_AGENTS)
R_centr = np.sum(R,axis=0)          # shape (D_AGENTS, )

# Optimum for the agents all together, for the agents in the "center"
X_opt = -np.linalg.inv(QQ_centr)@R_centr            # -R/D -> shape: (D_AGENTS, )
f_opt, _ = quad_funct(X_opt, QQ_centr, R_centr)     # shape ()
print(f'The optimal x is: {X_opt}')
print(f'The optimal cost is: {f_opt}')


# # Optimum for each agent
# X_opt = -np.linalg.inv(QQ)@R[..., None] #-R/D #shape (N, D, D)*(N, D, ) = (N_AGENTS, D_AGENTS , )
# print(np.transpose(X_opt, axes=(0, 2, 1)).shape, X_opt.shape, R[..., None].T.shape, X_opt.shape)
# F_opt = 0.5*np.transpose(X_opt, axes=(0, 2, 1))@QQ@X_opt+np.transpose(R[..., None], axes=(0, 2, 1))@X_opt #shape (D_AGENTS, )
# print(f'The optimal x is: {X_opt}')
# print(f'The optimal cost is: {np.sum(F_opt)}')
# print(f'The optimal cost is: {F_opt}')

###############################################################################
# GO!

for kk in range(MAXITERS-1):
    for ii in range(N_AGENTS):
        Nii = np.nonzero(ADJ[ii])[0]        # Find neighbours

        Cost[kk, ii], Grad[kk, ii] = quad_funct(X[kk, ii], QQ[ii], R[ii])
        
        # STATE UPDATE
        X[kk+1, ii] = WW[ii, ii] * X[kk, ii] - STEPSIZE * S[kk, ii]
        for jj in Nii:
            X[kk+1, ii] += WW[ii, jj] * X[kk, jj]
	    
        _, Grad[kk+1, ii] = quad_funct(X[kk+1, ii], QQ[ii], R[ii])
	
        # PROXY UPDATE
        S[kk+1, ii] = WW[ii, ii] * S[kk, ii] + Grad[kk+1, ii] - Grad[kk, ii]
        for jj in Nii:
            S[kk+1, ii] += WW[ii, jj] * S[kk, jj]

for ii in range(N_AGENTS):
    Cost[-1, ii], Grad[-1, ii] = quad_funct(X[-1, ii], QQ[ii], R[ii])


###############################################################################
############################	PLOTS	#######################################
###############################################################################

# Figure 1 : Cost Evolution
if 1:
	plt.figure()
	plt.plot(np.sum(Cost, axis=-1), label='Cost Evolution')
	plt.axhline(f_opt, color='r', linewidth=2, linestyle='--', label='f_opt')
	plt.xlim(0, MAXITERS)
	plt.xlabel(r"iterations $k$")
	plt.ylabel(r"$\sum_{i=1}^N f_i(x_i^k)$, $f^\star$")
	plt.title(r"Evolution of the cost")
	plt.legend()
	plt.grid()


###############################################################################
# Figure 2 : Cost Error Evolution
if 1:
	plt.figure()
	plt.semilogy(np.abs(np.sum(Cost, axis=-1)-f_opt), label='Cost Error Evolution')
	plt.xlim(0, MAXITERS)
	plt.xlabel(r"iterations $k$")
	plt.ylabel(r"$|\sum_{i=1}^N f_i(x_i^k) - f^\star|$")
	plt.title(r"Evolution of the cost error")
	plt.grid()


###############################################################################
# Figure 1 : Gradient Evolution
if 1:
	plt.figure()
	plt.semilogy(np.abs(np.sum(Grad, axis=1)))
	plt.xlim(0, MAXITERS)
	plt.xlabel(r"iterations $k$")
	plt.ylabel(r"DA FARE")
	plt.title(r"Evolution of the gradient for each dimension")
	plt.grid()

if 1:
    plt.figure()
    for d in range(D_AGENTS):
        plt.axhline(X_opt[d], color='r', linewidth=2, linestyle='--')
    #plt.plot(np.arange(MAXITERS), XX_avg, '--', linewidth=3)
    for ii in range(N_AGENTS):
        plt.plot(X[:, ii])
    	
    plt.xlabel(r"iterations $k$")
    plt.ylabel(r"$x^\star$, $\bar{x}^k$, $x_i^k$")
    plt.title(r"Evolution of the local estimates")
    plt.grid()
plt.show()