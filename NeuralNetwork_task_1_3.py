'''
20 may 2023
@ Giulia Cutini, Cenerini Simone, Riccardo Paolini

Multi-sample Neural-Network (Distributed Training)
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow.keras as ks
import pandas as pd
import networkx as nx

###############################################################################
# Set seed for reproducibility
SEED = 25
np.random.seed(SEED)

###############################################################################
# DataFrame Settings
TARGET = 3
SIZE = (4,4)
N_AGENTS = 5
SAMPLES_PER_AGENT = 6
SAMPLES = N_AGENTS*SAMPLES_PER_AGENT

# Load DataFrame
(x_train, y_train), (x_test, y_test) = ks.datasets.mnist.load_data()

# Preprocess the image, reshape & normalize
preprocess = lambda x: cv2.resize(x, SIZE).flatten() / 255.
x_train = [preprocess(x) for x in x_train]
x_test  = [preprocess(x) for x in x_test]

# Assign the TARGET
y_train = [1 if y == TARGET else 0 for y in y_train]
y_test  = [1 if y == TARGET else 0 for y in y_test]

# Select the SAMPLES
df_train = pd.DataFrame({'image': x_train, 'label': y_train}).groupby('label')
df_train_balanced = df_train.sample(SAMPLES//2, random_state=SEED).sample(frac=1, random_state=SEED)

images = np.array([df_train_balanced['image'][agent::N_AGENTS].tolist() for agent in range(N_AGENTS)])# [N_AGENTS, samples_per_agent, image_size]
labels = np.array([df_train_balanced['label'][agent::N_AGENTS].tolist() for agent in range(N_AGENTS)])# [N_AGENTS, samples_per_agent]

image_size = images.shape[-1]

print(f'Total positive samples {np.sum(labels == 1)}')
print(f'Total negative samples {np.sum(labels == 0)}')

###############################################################################
# Network setting
T_LAYERS = 3        # Number of layers
D_NEURONS = image_size      # Number of neurons for each layer
ActivationFunct = "Sigmoid" # {"Sigmoid", "ReLu", "HyTan"}
CostFunct = "Quadratic"     # {"Quadratic", "BinaryCrossEntropy"}

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

###############################################################################

# Cost Function
def cost_fn(Y,xT, mask=None):
    xT0 = xT[0] #It's a scalar

    if CostFunct == "BinaryCrossEntropy":
            J = -(Y*np.log(xT0) + 1e-10)-((1-Y)*(np.log(1-xT0)) + 1e-10)
            dJ = -Y/(xT0 + 1e-10) + (1-Y)/(1-xT0 + 1e-10)


    if CostFunct == "Quadratic":
            J = (xT0 - Y)*(xT0 - Y)
            dJ = 2*(xT0 - Y)

         
    if mask is not None:
            dJ = dJ*mask

    return J, dJ


# Activation Function
def activation_fn(xl):

    if ActivationFunct == "ReLu":       # Rectified Linear Unit
        out = max(0, xl) 

    if ActivationFunct == "HyTan":      # Hyperbolic tangent [-1, +1]
        out = np.tanh(xl)

    if ActivationFunct == "Sigmoid":    # Sigmoid function [0, +1]
        out = 1 / (1 + np.exp(-xl))

    return out


# Derivative of Activation Function
def activation_fn_derivative(xl):

    if ActivationFunct == "ReLu": # Rectified Linear Unit
        if xl > 0:
            out = 1
        else:
            out = 0

    if ActivationFunct == "HyTan": # Hyperbolic tangent
        out = 1-(activation_fn(xl))**2

    if ActivationFunct == "Sigmoid": # Sigmoid function
        out = activation_fn(xl)*(1-activation_fn(xl))

    return out


# Inference: xtp = f(xt,ut) (Activation Function for a whole layer)
def inference_dynamics(xt,ut):
    xtp = np.zeros(D_NEURONS)

    for l in range(D_NEURONS):
        temp = xt@ut[l,1:] + ut[l,0]
        xtp[l] = activation_fn(temp)

    return xtp
  

# Forward Propagation: (Inference dynamics for all the layers)
def forward_pass(x0, uu):
    xx = np.zeros((T_LAYERS, D_NEURONS))
    xx[0] = x0

    # Repeate the inference dynamics for all the layers
    for t in range(T_LAYERS-1):
        xx[t+1] = inference_dynamics(xx[t], uu[t])

    return xx
  

# Adjoint dynamics
def adjoint_dynamics(ltp, xt, ut):
    df_dx = np.zeros((D_NEURONS, D_NEURONS))
    df_du = np.zeros(((D_NEURONS+1)*D_NEURONS, D_NEURONS))

    dim = np.tile([D_NEURONS+1], D_NEURONS)
    cs_idx = np.append(0, np.cumsum(dim))
  
    for l in range(D_NEURONS):
        xl = xt@ut[l, 1:] + ut[l, 0]
        dSigma = activation_fn_derivative(xl)

        df_dx[:, l] = dSigma*ut[l, 1:] # A matrix
        df_du[cs_idx[l]:cs_idx[l+1], l] = dSigma * np.hstack([1, xt])

    lt = df_dx @ ltp # Adjoint equation
    delta_ut_vec = df_du @ ltp
    delta_ut = np.reshape(delta_ut_vec,(D_NEURONS, D_NEURONS+1))

    return lt, delta_ut


# Backward Propagation: (Adjoint dynamics for all the layers)
def backward_pass(xx, uu, llambdaT):
    llambda = np.zeros((T_LAYERS, D_NEURONS))
    delta_u = np.zeros((T_LAYERS-1, D_NEURONS, D_NEURONS+1))
    llambda[-1] = llambdaT

    for t in reversed(range(T_LAYERS-1)):
        llambda[t], delta_u[t] = adjoint_dynamics(llambda[t+1], xx[t], uu[t])

    return llambda, delta_u

# Computes the number of correctly and wrong classified samples
def accuracy(xT,Y):
    error = 0
    success = 0

    if xT>0.5:
        if Y==0: # Missclassified
            error += 1
        else:
            success += 1
    else:
        if Y==0:  #Correctly classified
            success += 1
        else:
            error +=1 

    return success, error



###############################################################################
# MAIN
###############################################################################

# Training parameters
EPOCHS = 10000
STEP_SIZE = 1e-1
BATCH_SIZE = 2 # Dimension of the minibatch set
N_BATCH = int(np.ceil(SAMPLES_PER_AGENT/BATCH_SIZE))

# Network Variables
xx = np.zeros((N_AGENTS, BATCH_SIZE, T_LAYERS, D_NEURONS))
weight_init = np.random.randn(T_LAYERS-1, D_NEURONS, D_NEURONS+1)*1e-2
uu = np.array([weight_init for _ in range(N_AGENTS)])
vv = np.zeros((N_AGENTS, T_LAYERS-1, D_NEURONS, D_NEURONS+1))
#ss = np.zeros((N_AGENTS, T_LAYERS-1, D_NEURONS, D_NEURONS+1))

# Mask for output regression
mask = np.zeros(D_NEURONS)
mask[0] = 1

J = np.zeros((EPOCHS, N_AGENTS)) # Cost function
NormGradientJ = np.zeros((EPOCHS, N_AGENTS))

# Initialization for Accuracy
successes = 0
errors = 0

for epoch in range(EPOCHS):
    if epoch % 10 == 0 and epoch != 0:
        print(f'Cost at k={epoch:d} is {np.mean(J[epoch-1]):.4f}')

    for batch_num in range(N_BATCH):
        for agent in range(N_AGENTS):
            neighs = np.nonzero(ADJ[agent])[0]

            # Distributed Gradient Algorithm
            vv[agent] = WW[agent, agent] * uu[agent] 
            for neigh in neighs:
                vv[agent] += WW[agent, neigh] * uu[neigh]

            batch_grad = 0
            for batch_el in range(BATCH_SIZE):
                idx = (batch_num*BATCH_SIZE) + batch_el
                
                # Skip if samples are finished (last minibatch)
                if idx >= SAMPLES_PER_AGENT:
                    break

                xx[agent, batch_el] = forward_pass(images[agent, idx], vv[agent])
                pred = xx[agent, batch_el, -1, :]
                loss, out_grad = cost_fn(labels[agent, idx], pred, mask)
                _, grad = backward_pass(xx[agent, batch_el], vv[agent], out_grad) # out_grad = llambdaT

                J[epoch, agent] += loss / SAMPLES_PER_AGENT
                batch_grad += grad / BATCH_SIZE

            vv[agent] = vv[agent] - (STEP_SIZE * batch_grad)
            NormGradientJ[epoch, agent] += np.linalg.norm(batch_grad) / N_BATCH
        
        # SYNCHRONOUS UPDATE
        for agent in range(N_AGENTS): 
            uu[agent] = vv[agent]

# for img in range(BATCH_SIZE):
#     for agents in range(N_AGENTS):
#         print(f"Label for Image {img} was {labels[agent,img]} but is classified as:", xx[img,-1, 0])


# for agents in range(N_AGENTS):
#     for img in range(BATCH_SIZE):
#         idx = ((N_BATCH-1)*BATCH_SIZE) + img
#         print(f"Label for Image {idx} was {labels[idx]} but is classified as:", xx[img,-1, 0])


# ###############################################################################
# # Accuracy computation
# for img in range(BATCH_SIZE):
#     print(f"Label for Image {img} was {labels[img]} but is classified as:", xx[img,-1, 0])
#     success, error = accuracy(xx[img,-1, 0],labels[img])
#     successes += success
#     errors += error

# percentage_of_success = (successes/(successes+errors))*100
# print("Correctly classified point: ", successes)
# print("Wrong classified point: ", errors)
# print("Percentage of Success: ", percentage_of_success)

                
