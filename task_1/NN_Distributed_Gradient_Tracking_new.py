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
from copy import deepcopy
import os
import pickle

###############################################################################
# Set seed for reproducibility
SEED = 25
np.random.seed(SEED)

save_weights = True

###############################################################################
# DataFrame Settings
TARGET = 8
SIZE = (28, 28)
N_AGENTS = 5
SAMPLES_PER_AGENT = 256 # Multiple of Minibatch Size 
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
T_LAYERS = 2        # Number of layers
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
print('Col Stochasticity {}\n'.format(np.sum(WW,axis=0)))

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
def inference_dynamics(xt, ut):
    n_neurons = ut.shape[0]
    xtp = np.zeros(n_neurons)

    for n in range(n_neurons):
        temp = xt@ut[n, 1:] + ut[n, 0] #temp -> scalar
        xtp[n] = activation_fn(temp)

    return xtp
  

# Forward Propagation: (Inference dynamics for all the layers)
def forward_pass(x0, uu_list):
    xx = [x0]

    # Repeate the inference dynamics for all the layers
    for l_idx, uu in enumerate(uu_list):
        xx.append(inference_dynamics(xx[l_idx], uu))
        
    return xx
  

# Adjoint dynamics
def adjoint_dynamics(llambda_tp, xt, ut):
    out_neurons = ut.shape[0]
    in_neurons = ut.shape[1] - 1 # '-1' for the bias
    df_dx = np.zeros((in_neurons, out_neurons))
    df_du = np.zeros(((in_neurons+1)*out_neurons, out_neurons))

    dim = np.tile([in_neurons+1], out_neurons)
    cs_idx = np.append(0, np.cumsum(dim))
    for n in range(out_neurons):
        xtp = xt@ut[n, 1:] + ut[n, 0]
        dSigma = activation_fn_derivative(xtp)

        df_dx[:, n] = dSigma*ut[n, 1:] # A matrix
        df_du[cs_idx[n]:cs_idx[n+1], n] = dSigma * np.hstack([1, xt])

    llambda_t = df_dx @ llambda_tp # Adjoint equation
    delta_ut_vec = df_du @ llambda_tp
    delta_ut = np.reshape(delta_ut_vec,(out_neurons, in_neurons+1))

    return llambda_t, delta_ut


# Backward Propagation: (Adjoint dynamics for all the layers)
def backward_pass(xx, uu, llambdaT):
    '''
    Input:
        xx = [x_0, ..., x_n]        list of vectors
        uu = [u_0, ..., u_(n-1)]    list of matrices
        llambdaT.shape == x_n.shape

    Return:
        llambda = [lambda_0, ..., lambda_n]         list of vectors
        delta_u = [delta_u_0, ..., delta_u_(n-1)]   list of matrices

    '''
    llambda = [np.zeros_like(xt) for xt in xx] # llambda[of x_0] is never used
    delta_u = [np.zeros_like(ut) for ut in uu]
    llambda[-1] = llambdaT

    n_layers = len(xx)
    for l_idx in reversed(range(n_layers-1)):
        llambda[l_idx], delta_u[l_idx] = adjoint_dynamics(llambda[l_idx+1], xx[l_idx], uu[l_idx])

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
EPOCHS = 1000
STEP_SIZE = 1e-1
BATCH_SIZE = 8 # Dimension of the minibatch set
N_BATCH = int(np.ceil(SAMPLES_PER_AGENT/BATCH_SIZE))

# Network Variables
# network = [28*28, 28, 1]
network = [(SIZE[0]*SIZE[1]), int(np.sqrt(SIZE[0]*SIZE[1])) , 1]
n_layers = len(network)
# xx = [np.zeros(shape=(n_neurons,)) for n_neurons in network] # shape[network.shape]
uu = [np.random.randn(network[layer_idx+1], network[layer_idx]+1) for layer_idx in range(len(network)-1)]
ss = [np.zeros_like(ul) for ul in uu]
old_grad = [np.zeros_like(ul) for ul in uu]


uu = [[deepcopy(uu) for _ in range(N_AGENTS)] for _ in range(N_BATCH*EPOCHS+1)] # shape[EPOCHS*N_BATCH, N_AGENTS, weights.shape]
ss = [[deepcopy(ss) for _ in range(N_AGENTS)] for _ in range(N_BATCH*(EPOCHS+1))] # shape[EPOCHS*N_BATCH, N_AGENTS, weights.shape]
old_grad = [[deepcopy(old_grad) for _ in range(N_AGENTS)] for _ in range(N_BATCH)] # shape[N_BATCH, N_AGENTS, weights.shape]

prediction = np.zeros((N_AGENTS, SAMPLES))

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
for batch_num in range(N_BATCH):
    for agent in range(N_AGENTS):
        for batch_el in range(BATCH_SIZE):
            idx = (batch_num*BATCH_SIZE) + batch_el

            # Skip if samples are finished (last minibatch)
            if idx >= SAMPLES_PER_AGENT:
                break
            
            # Forward pass
            xx = forward_pass(images_train[agent, idx], uu[0][agent])
            #print(xx[-1])
            # Loss evaluation
            loss, out_grad = cost_fn(labels_train[agent, idx], xx[-1])

            # Backward pass
            _, grad = backward_pass(xx, uu[0][agent], out_grad) # out_grad = llambdaT

            # Gradient accumulation
            for layer in range(n_layers-1):
                old_grad[batch_num][agent][layer] += grad[layer] / BATCH_SIZE

        for layer in range(n_layers-1):
            ss[batch_num][agent][layer] = old_grad[batch_num][agent][layer]

###############################################################################
# Training
for epoch in range(EPOCHS):
    # Early stopping if training is not leading to better results
    if epoch >= 11 and np.all([np.mean(J[e]) > np.mean(J[epoch-11]) - 0.001 for e in range(epoch-10, epoch)]):
        break

    if epoch % 1 == 0 and epoch != 0:
        print(f'[k={epoch:d}] Cost is {np.mean(J[epoch-1]):.4f} and Grandient is {np.mean(NormGradientJ[epoch-1]):.4f}')

    for batch_num in range(N_BATCH):
        kk = epoch*N_BATCH+batch_num

        for agent in range(N_AGENTS):
            neighs = np.nonzero(ADJ[agent])[0]

            # Gradient Tracking Algorithm - Weights Update
            for layer in range(n_layers-1):
                uu[kk+1][agent][layer] = (WW[agent, agent] * uu[kk][agent][layer]) - (STEP_SIZE * ss[kk][agent][layer])
                for neigh in neighs:
                    uu[kk+1][agent][layer] += WW[agent, neigh] * uu[kk][neigh][layer]

            batch_grad = [np.zeros_like(ul) for ul in uu[kk][agent]]
            for batch_el in range(BATCH_SIZE):
                idx = (batch_num*BATCH_SIZE) + batch_el
                
                # Skip if samples are finished (last minibatch)
                if idx >= SAMPLES_PER_AGENT:
                    break

                # Forward pass
                xx = forward_pass(images_train[agent, idx], uu[kk+1][agent])
                prediction[agent, idx] = xx[-1] # prediction <= value of the first neuron in the last layer
                
                # Loss evalutation
                loss, out_grad = cost_fn(labels_train[agent, idx], xx[-1])

                # Backward pass
                _, grad = backward_pass(xx, uu[kk+1][agent], out_grad) # out_grad = llambdaT

                J[epoch, agent] += loss / SAMPLES_PER_AGENT
                for layer in range(n_layers-1):
                    batch_grad[layer] += grad[layer] / BATCH_SIZE

            for layer in range(n_layers-1):
                NormGradientJ[epoch, agent] += np.linalg.norm(batch_grad[layer] / batch_grad[layer].size) / N_BATCH
            
            # Gradient Tracking Algorithm - SS Update
            for layer in range(n_layers-1):
                ss[kk+N_BATCH][agent][layer] = (WW[agent, agent] * ss[kk][agent][layer]) + (batch_grad[layer] - old_grad[batch_num][agent][layer])
                for neigh in neighs:
                    ss[kk+N_BATCH][agent][layer] += WW[agent, neigh] * ss[kk][neigh][layer]

            for layer in range(n_layers-1):
                old_grad[batch_num][agent][layer] = batch_grad[layer] 

print('\n\nTRAINING SET\n')
for agent in range(N_AGENTS):
    for batch_el in range(BATCH_SIZE):
        idx = ((N_BATCH-1)*BATCH_SIZE) + batch_el
        if idx >= SAMPLES_PER_AGENT:    # Skip if samples are finished (last minibatch)
            break
        print(f"[Agent {agent}] Label for Image {idx} was {labels_train[agent,idx]} and is classified as {prediction[agent,idx]:.4f}")
    print()

print('\n\nTEST SET\n')
for agent in range(N_AGENTS):
    for batch_el in range(BATCH_SIZE):
        idx = ((N_BATCH-1)*BATCH_SIZE) + batch_el
        if idx >= SAMPLES_PER_AGENT:    # Skip if samples are finished (last minibatch)
            break
        print(f"[Agent {agent}] Label for the SAME Image {idx} was {labels_test[0, idx]} and is classified as {forward_pass(images_test[0, idx], uu[-1][agent])[-1][0]:.4f}")
    print()


# ###############################################################################
# Accuracy computation
print('\n--------------TRAINING SCORES----------------')
for agent in range(N_AGENTS):
    for img in range(SAMPLES_PER_AGENT):
        success, error = accuracy(prediction[agent, img], labels_train[agent, img])
        successes_train[agent] += success 
        errors_train[agent] += error

    percentage_of_success_train[agent] = (successes_train[agent] / (SAMPLES_PER_AGENT)) * 100
    print('\nAGENT: ', agent)
    print("Correctly classified point: ", successes_train[agent])
    print("Wrong classified point: ", errors_train[agent])
    print(f"Accuracy: {percentage_of_success_train[agent]:.4f}" )  

print('\n------------------TEST SCORES-----------------')
for agent in range(N_AGENTS):
    for img in range(SAMPLES_PER_AGENT):
        output = forward_pass(images_test[0, img], uu[-1][agent])
        success, error = accuracy(output[-1][0], labels_test[0, img])
        successes_test[agent] += success
        errors_test[agent] += error

    percentage_of_success_test[agent] = (successes_test[agent] / (SAMPLES_PER_AGENT)) * 100
    print('\nAGENT: ', agent)
    print("Correctly classified point: ", successes_test[agent])
    print("Wrong classified point: ", errors_test[agent])
    print(f"Accuracy: {percentage_of_success_test[agent]:.4f}" ) 

###############################################################################
# Save weights of agent 0 (theoretically at consensous)
if save_weights:
    weights_file = f'Grad_Track-weights_{SIZE[0]}x{SIZE[1]}_E{EPOCHS}_S{SAMPLES}_B{BATCH_SIZE}.pkl'
    weights_path = os.path.join(os.getcwd(), 'task_1/weights', weights_file)
    with open(weights_path, 'wb') as f:
        pickle.dump(uu[-1][0], f)
                
# ###############################################################################
# # PLOT
# ###############################################################################

plt.plot()

plt.figure('Cost function')
plt.plot(range(EPOCHS), np.mean(J, axis=1), label='Total Normalized Cost Evolution', linewidth = 3)
for agent in range(N_AGENTS):
     plt.plot(range(EPOCHS), J[:, agent], linestyle = ':')
plt.xlabel(r'Epochs')
plt.ylabel(r'J')
plt.legend()
plt.title('J')
plt.grid()


plt.figure('Norm of Cost function')
plt.semilogy(range(EPOCHS), np.mean(NormGradientJ, axis=-1), label='Total Gradient Evolution', linewidth = 3)
for agent in range(N_AGENTS):
    plt.semilogy(range(EPOCHS), NormGradientJ[:, agent], linestyle = ':')
plt.xlabel(r'Epochs')
plt.legend()
plt.title('norm_gradient_J')
plt.grid()


# Computes the mean error over uu
plt.figure('UU error')
uu_mean = [np.mean([np.sum([np.sum(np.abs(uu[t][agent][layer])) / uu[t][agent][layer].size for layer in range(n_layers-1)]) for agent in range(N_AGENTS)]) for t in range(N_BATCH*EPOCHS+1)]
#plt.plot([uu_mean[t] for t in range(N_BATCH*EPOCHS+1)])
for agent in range(N_AGENTS):
    plt.plot([(np.sum([np.sum(np.abs(uu[t][agent][layer])) / uu[t][agent][layer].size for layer in range(n_layers-1)]) - uu_mean[t]) / uu_mean[t] for t in range(0, N_BATCH*EPOCHS+1, N_BATCH)], linestyle=':')
plt.xlabel(r'Epochs')
plt.title('UU magnitude - UU_mean magnitude (scaled)')
plt.grid()


# plt.figure('UU evolution')
# plt.plot([uu_mean[t] for t in range(0, N_BATCH*(EPOCHS//10)+1, N_BATCH)], label='UU mean', linewidth = 2)
# for agent in range(N_AGENTS):
#     plt.plot([np.sum([np.sum(np.abs(uu[t][agent][layer])) / uu[t][agent][layer].size for layer in range(n_layers-1)]) for t in range(0, N_BATCH*(EPOCHS//10)+1, N_BATCH)], linestyle=':')
# plt.xlabel(r'Epochs')
# plt.title('UU for the first 10% of epochs')
# plt.grid()


plt.figure('Wheight of single connection')
uu_single_mean = [np.mean([np.sum(np.abs(uu[t][agent][-1][0, 1])) for agent in range(N_AGENTS)]) for t in range(N_BATCH*EPOCHS+1)]
for agent in range(N_AGENTS):
    plt.plot([(uu[t][agent][-1][0, 1] - uu_single_mean[t]) / uu_single_mean[t] for t in range(0, N_BATCH*EPOCHS+1, N_BATCH)], linewidth = 0.5)
plt.xlabel(r'Epochs')
plt.title('Difference between the wheight of last neurons and its mean value (scaled)')
plt.grid()


plt.figure('SS evolution')
plt.semilogy([np.mean([np.sum([np.sum([np.sum(np.abs(ss[e*N_BATCH+b][agent][layer])) / ss[e*N_BATCH+b][agent][layer].size for layer in range(n_layers-1)]) for b in range(N_BATCH)]) for agent in range(N_AGENTS)]) for e in range(EPOCHS)], label='SS mean', linewidth = 3)
for agent in range(N_AGENTS):
    plt.semilogy([np.sum([np.sum([np.sum(np.abs(ss[e*N_BATCH+b][agent][layer])) / ss[e*N_BATCH+b][agent][layer].size for layer in range(n_layers-1)]) for b in range(N_BATCH)]) for e in range(EPOCHS)], linestyle = ':')
plt.xlabel(r'Epochs')
plt.legend()
plt.title('SS magnitude')
plt.grid()
plt.show()


# shape[EPOCHS*N_BATCH, N_AGENTS, weights.shape]
# plt.figure('uu of first neuron of second layer')
# plt.plot([uu[t][agent][1,]])

# plt.figure('UU (Weights) evolution')
# plt.plot(range(EPOCHS*N_BATCH), np.mean(uu, axis=(1,2)), label='Total UU Evolution', linewidth = 3)
# for agent in range(N_AGENTS):
#     plt.plot(range(EPOCHS*N_BATCH), np.mean(uu, axis=2)[:][agent], linestyle = ':')
# plt.xlabel(r'Epochs')
# plt.legend()
# plt.title('SS')
# plt.grid()

# plt.show()


# # plt.figure('UU (Weights) evolution')
# # plt.plot(range(EPOCHS), np.mean(uu_mean, axis=1), label='Total UU Evolution', linewidth = 3)
# # for agent in range(N_AGENTS):
# #     plt.plot(range(EPOCHS), uu_mean[:, agent], linestyle = ':')
# # plt.xlabel(r'Epochs')
# # plt.legend()
# # plt.title('UU')
# # plt.grid()