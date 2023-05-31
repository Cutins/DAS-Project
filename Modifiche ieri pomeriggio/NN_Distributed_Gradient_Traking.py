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
import os
import pickle

###############################################################################
# Set seed for reproducibility
SEED = 25
np.random.seed(SEED)

save_weights = True

###############################################################################
# DataFrame Settings
TARGET = 3
SIZE = (15, 15)
N_AGENTS = 5
SAMPLES_PER_AGENT = 10
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

# Balance the Dataset, select the SAMPLES
df_train = pd.DataFrame({'image': x_train, 'label': y_train}).groupby('label')
df_train_balanced = df_train.sample(SAMPLES, random_state=SEED).sample(frac=1, random_state=SEED)

images_train = np.array([df_train_balanced['image'][agent:SAMPLES:N_AGENTS].tolist() for agent in range(N_AGENTS)])# [N_AGENTS, samples_per_agent, image_size]
labels_train = np.array([df_train_balanced['label'][agent:SAMPLES:N_AGENTS].tolist() for agent in range(N_AGENTS)])# [N_AGENTS, samples_per_agent]

images_test = np.array([df_train_balanced['image'][agent+SAMPLES::N_AGENTS].tolist() for agent in range(N_AGENTS)])
labels_test = np.array([df_train_balanced['label'][agent+SAMPLES::N_AGENTS].tolist() for agent in range(N_AGENTS)])

image_size = images_train.shape[-1]

print(f'Total positive samples {np.sum(labels_train == 1)}')
print(f'Total negative samples {np.sum(labels_train == 0)}')

###############################################################################
# Network setting
T_LAYERS = 4        # Number of layers
D_NEURONS = image_size      # Number of neurons for each layer
ActivationFunct = "Sigmoid" # {"Sigmoid", "ReLu", "HyTan"}
CostFunct = "BinaryCrossEntropy"     # {"Quadratic", "BinaryCrossEntropy"}

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

# plt.show()s

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

print('Row Stochasticity {}'.format(np.sum(WW,axis=1)))
print('Col Stochasticity {}'.format(np.sum(WW,axis=0)))

###############################################################################

# Cost Function
def cost_fn(Y,xT0):
    '''
    Y [Scalar]
    XT0 [Scalar]
    '''

    if CostFunct == "BinaryCrossEntropy":
            J = -(Y*np.log(xT0 + 1e-5) )-((1-Y)*(np.log(1-xT0 + 1e-5)))
            dJ = -Y/(xT0 + 1e-5) + (1-Y)/(1-xT0 + 1e-5)


    if CostFunct == "Quadratic":
            J = (xT0 - Y)*(xT0 - Y)
            dJ = 2*(xT0 - Y)

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
EPOCHS = 300
STEP_SIZE = 1e-1
BATCH_SIZE = 8 # Dimension of the minibatch set
N_BATCH = int(np.ceil(SAMPLES_PER_AGENT/BATCH_SIZE))

# Network Variables
xx = np.zeros((T_LAYERS, D_NEURONS))
weight_init = np.random.randn(T_LAYERS-1, D_NEURONS, D_NEURONS+1)*1e-2
uu = np.array([weight_init for _ in range(N_AGENTS)])
uu_plus = np.zeros((N_AGENTS, T_LAYERS-1, D_NEURONS, D_NEURONS+1))
ss = np.zeros((N_AGENTS, T_LAYERS-1, D_NEURONS, D_NEURONS+1))
ss_plus = np.zeros((N_AGENTS, T_LAYERS-1, D_NEURONS, D_NEURONS+1))
batch_grad = np.zeros((N_AGENTS, T_LAYERS-1, D_NEURONS, D_NEURONS+1))
prediction = np.zeros((N_AGENTS,SAMPLES))

uu_ELIMINA = np.zeros((EPOCHS, N_AGENTS, T_LAYERS-1, D_NEURONS, D_NEURONS+1))
ss_ELIMINA = np.zeros((EPOCHS, N_AGENTS, T_LAYERS-1, D_NEURONS, D_NEURONS+1))

J = np.zeros((EPOCHS, N_AGENTS)) # Cost function
NormGradientJ = np.zeros((EPOCHS, N_AGENTS))

# Initialization for Accuracy
successes_train = np.zeros((N_AGENTS))
errors_train = np.zeros((N_AGENTS))
percentage_of_success_train = np.zeros((N_AGENTS))
successes_test = np.zeros((N_AGENTS))
errors_test = np.zeros((N_AGENTS))
percentage_of_success_test = np.zeros((N_AGENTS))

###############################################################################
# Initialization of Gradient Tracking Algorithm
for agent in range(N_AGENTS):
    batch_grad[agent] = 0
    for idx in range(BATCH_SIZE):
        
        # Forward pass
        xx = forward_pass(images_train[agent, idx], uu[agent])

        # Loss evaluation
        out_grad = np.zeros((D_NEURONS)) # Initialize output gradient to 0 (for output regression)out_grad
        loss, out_grad[0] = cost_fn(labels_train[agent, idx], xx[-1, 0])

        # Backward pass
        _, grad = backward_pass(xx, uu[agent], out_grad) # out_grad = llambdaT

        # Gradient accumulation
        batch_grad[agent] += grad / BATCH_SIZE
        
    ss[agent] = batch_grad[agent]
        
###############################################################################
# Training
for epoch in range(EPOCHS):
    if epoch % 1 == 0 and epoch != 0:
        print(f'Cost at k={epoch:d} is {np.mean(J[epoch-1]):.4f} and Grandient is {np.mean(NormGradientJ[epoch-1]):.4f}')

    for batch_num in range(N_BATCH):
        for agent in range(N_AGENTS):
                
            neighs = np.nonzero(ADJ[agent])[0]

            # Gradient Tracking Algorithm - Weights Update
            uu_plus[agent] = (WW[agent, agent] * uu[agent]) - (STEP_SIZE * ss[agent])
            for neigh in neighs:
                uu_plus[agent] += WW[agent, neigh] * uu[neigh]

            batch_grad_plus = 0
            for batch_el in range(BATCH_SIZE):
                idx = (batch_num*BATCH_SIZE) + batch_el
                
                # Skip if samples are finished (last minibatch)
                if idx >= SAMPLES_PER_AGENT:
                    break

                # Forward pass
                xx = forward_pass(images_train[agent, idx], uu_plus[agent])
                prediction[agent,idx] = xx[-1, 0] # prediction <= value of the first neuron in the last layer

                # Loss evalutation
                out_grad = np.zeros((D_NEURONS)) # Initialize output gradient to 0 (for output regression)
                loss, out_grad[0] = cost_fn(labels_train[agent, idx], prediction[agent,idx])

                # Backward pass
                _, grad = backward_pass(xx, uu_plus[agent], out_grad) # out_grad = llambdaT

                J[epoch, agent] += loss / SAMPLES_PER_AGENT
                batch_grad_plus += grad / BATCH_SIZE

            NormGradientJ[epoch, agent] += np.linalg.norm(batch_grad_plus) / N_BATCH

            # Gradient Tracking Algorithm - SS Update
            ss_plus[agent] = (WW[agent, agent] * ss[agent]) + (batch_grad_plus - batch_grad[agent])
            for neigh in neighs:
                ss_plus[agent] += WW[agent, neigh] * ss[neigh]
            
            batch_grad[agent] = batch_grad_plus
        
        # SYNCHRONOUS UPDATE
        for agent in range(N_AGENTS): 
            uu[agent] = uu_plus[agent]
            ss[agent] = ss_plus[agent]

            ss_ELIMINA[epoch,agent] = ss[agent]     # ELIMINAMI
            uu_ELIMINA[epoch, agent] = uu[agent]    # ELIMINAMI

# Computes the mean error over uu
uu_mean = np.mean(uu, axis=0)
for agent in range(N_AGENTS):
    print(f'The Agent {agent} has mean error = {np.linalg.norm(uu_mean - uu[agent])}')


print('\n\nTRAINING SET\n')
for agent in range(N_AGENTS):
    for batch_el in range(BATCH_SIZE):
        idx = ((N_BATCH-1)*BATCH_SIZE) + batch_el
        print(f"[Agent {agent}] Label for Image {idx} was {labels_train[agent,idx]} but is classified as :{prediction[agent,idx]:.4f}")
    print()

print('\n\nTEST SET\n')
for agents in range(N_AGENTS):
    for batch_el in range(BATCH_SIZE):
        idx = ((N_BATCH-1)*BATCH_SIZE) + batch_el
        print(f"[Agent {agents}] Label for the SAME Image {idx} was {labels_test[0, idx]} but is classified as {forward_pass(images_test[0, idx], uu[agent])[-1,0]:.4f}")
    print()


# ###############################################################################
# Accuracy computation
print('\n--------------TRAINING SCORES----------------')
for agent in range(N_AGENTS):
    for img in range(SAMPLES_PER_AGENT):
        success, error = accuracy(prediction[agent,img],labels_train[agent,img])
        successes_train[agent] += success
        errors_train[agent] += error

    percentage_of_success_train[agent] = (successes_train[agent]/(SAMPLES_PER_AGENT))*100
    print('\nAGENT: ', agent)
    print("Correctly classified point: ", successes_train[agent])
    print("Wrong classified point: ", errors_train[agent])
    print(f"Accuracy: {percentage_of_success_train[agent]:.4f}" )  

print('\n------------------TEST SCORES-----------------')
for agent in range(N_AGENTS):
    for img in range(SAMPLES_PER_AGENT):
        output = forward_pass(images_test[0,img], uu[agent])
        success, error = accuracy(output[-1, 0], labels_test[0,img])
        successes_test[agent] += success
        errors_test[agent] += error

    percentage_of_success_test[agent] = (successes_test[agent]/(SAMPLES_PER_AGENT))*100
    print('\nAGENT: ', agent)
    print("Correctly classified point: ", successes_test[agent])
    print("Wrong classified point: ", errors_test[agent])
    print(f"Accuracy: {percentage_of_success_test[agent]:.4f}" ) 

###############################################################################
# Save weights of agent 0 (theoretically at consensous)
if save_weights:
    weights_file = f'Grad_Track-weights_{SIZE[0]}x{SIZE[1]}_E{EPOCHS}_S{SAMPLES}_B{BATCH_SIZE}.pkl'
    weights_path = os.path.join(os.getcwd(), 'weights', weights_file)
    with open(weights_path, 'wb') as f:
        pickle.dump(uu[0], f)
                
###############################################################################
# PLOT
###############################################################################
plt.figure('Cost function')
plt.plot(range(EPOCHS),np.sum(J, axis=1)/N_AGENTS, label='Total Normalized Cost Evolution', linewidth = 3)
for agent in range(N_AGENTS):
     plt.plot(range(EPOCHS), J[:, agent], linestyle = ':')
plt.xlabel(r'Epochs')
plt.ylabel(r'J')
plt.legend()
plt.title('J')
plt.grid()

plt.figure('Norm of Cost function')
plt.semilogy(range(EPOCHS), np.sum(NormGradientJ, axis=-1)/N_AGENTS, label='Total Gradient Evolution', linewidth = 3)
for agent in range(N_AGENTS):
    plt.semilogy(range(EPOCHS), NormGradientJ[:, agent], linestyle = ':')
plt.xlabel(r'Epochs')
plt.legend()
plt.title('norm_gradient_J')
plt.grid()

# ss_ELIMINA = np.zeros((EPOCHS, N_AGENTS, T_LAYERS-1, D_NEURONS, D_NEURONS+1))
mean_along_layers_ss = np.mean(ss_ELIMINA, axis=2)
mean_along_neurons_ss = np.mean(mean_along_layers_ss, axis=(2,3))
ss_mean = mean_along_neurons_ss

mean_along_layers_uu = np.mean(uu_ELIMINA, axis=2)
mean_along_neurons_uu = np.mean(mean_along_layers_uu, axis=(2,3))
uu_mean = mean_along_neurons_uu

plt.figure('SS evolution')
plt.semilogy(range(EPOCHS), np.mean(ss_mean, axis=1), label='Total SS Evolution', linewidth = 3)
for agent in range(N_AGENTS):
    plt.semilogy(range(EPOCHS), ss_mean[:, agent], linestyle = ':')
plt.xlabel(r'Epochs')
plt.legend()
plt.title('SS')
plt.grid()

plt.figure('UU (Weights) evolution')
plt.plot(range(EPOCHS), np.mean(uu_mean, axis=1), label='Total UU Evolution', linewidth = 3)
for agent in range(N_AGENTS):
    plt.plot(range(EPOCHS), uu_mean[:, agent], linestyle = ':')
plt.xlabel(r'Epochs')
plt.legend()
plt.title('UU')
plt.grid()

plt.show()