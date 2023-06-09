#
# Containment Algorithm DT Time-varying
# Lorenzo Pichierri, Ivano Notarnicola
# Bologna, 03/04/2023
#
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from functions import waves, animation

np.random.seed(5)

# Bool vars
ANIMATION = True

#################################################################

TT = 60.0	  # Simulation time
NN = 20     # number of agents
n_x = 2     # dimension of x_i 
n_leaders = 4

dt = 0.01 	# Sampling time
horizon = np.arange(0.0, TT, dt)

p_ER = 0.1

I_NN = np.eye(NN)
I_nx = np.eye(n_x)
I_NN_nx = np.eye(n_x*NN)
O_NN = np.ones((NN,1))

# Let consider the whole state, included the integral dynamics (-> NN*n_x*2)
XX = np.zeros((n_x*NN*2,len(horizon)))

# Initiate the agents' state (keep integral init to zero)
XX_init = np.zeros((n_x*NN*2,1))
XX_init[0:n_x*NN] += 5*np.random.rand(n_x*NN,1)

XX[:,0] = XX_init.T

# Leaders input: null, sinusoidal
(amp, omega, phi) = (2, 0.1, 0)
UU = waves(amp, omega, phi, horizon, n_x, n_leaders)
# UU = np.zeros((n_leaders*n_x, len(horizon)))
UU = 10 * np.ones((n_leaders*n_x, len(horizon)))


for tt in range(len(horizon)-1):
  if tt%200:
    print('Iteration ',tt)

  # Generate new graph for each iteration (p_ER low)
  G = nx.binomial_graph(NN,p_ER)
  Adj = nx.adjacency_matrix(G).toarray()

  # Do not need connectivity for each t -> ... why?

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

  # Consider only the leaders in the B matrix
  BB_kron = np.zeros((NN*n_x,n_leaders*n_x))
  BB_kron[(NN-n_leaders)*n_x:,:] = np.identity(n_x*n_leaders, dtype=int)

  ################################################
  ## followers integral Action  
  
  k_i = 0 #20
  K_I = - k_i*I_NN_nx

  # Setup the extended dynamics
  LL_ext_up = np.concatenate((LL_kron, K_I), axis = 1)
  LL_ext_low = np.concatenate((LL_kron, np.zeros(LL_kron.shape)), axis = 1)
  LL_ext = np.concatenate((LL_ext_up, LL_ext_low), axis = 0)

  # extende BB_kron with integral state (null)
  BB_kron = np.concatenate((BB_kron, np.zeros((NN*n_x,n_leaders*n_x))), axis = 0)

  A = -LL_ext
  B = BB_kron

  ################################################
  # CONTAINMENT Dynamics

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
    plt.figure()
    animation(XX, NN, n_x, n_leaders, horizon, dt = 100)
  else:
    print('Animation allowed only for bi-dimensional agent')

plt.show()
