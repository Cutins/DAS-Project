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
from lib.graph import *

###############################################################################
# Set seed for reproducibility
SEED = 25
np.random.seed(SEED)

###############################################################################
# DataFrame Settings
TARGET = 3
SIZE = (4, 4)
N_AGENTS = 5
SAMPLES_PER_AGENT = 80
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

G, ADJ = get_graph(GRAPH_TYPE, N_AGENTS)
WW = get_weight_matrix(ADJ) # METROPOLIS HASTING

###############################################################################
# Network Function
if 1:
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
EPOCHS = 100
STEP_SIZE = 1
BATCH_SIZE = 8 # Dimension of the minibatch set
N_BATCH = int(np.ceil(SAMPLES_PER_AGENT/BATCH_SIZE))

# Network Variables
xx = np.zeros((T_LAYERS, D_NEURONS))

weight_init = np.random.randn(T_LAYERS-1, D_NEURONS, D_NEURONS+1)*1e-2
uu = np.array([weight_init for _ in range(N_AGENTS)])
uu_plus = np.zeros_like(uu)

ss = np.zeros((N_AGENTS, T_LAYERS-1, D_NEURONS, D_NEURONS+1))
ss_plus = np.zeros_like(ss)

batch_grad = np.zeros((N_AGENTS, T_LAYERS-1, D_NEURONS, D_NEURONS+1))
prediction = np.zeros((N_AGENTS,SAMPLES))

J = np.zeros((EPOCHS, N_AGENTS)) # Cost function
NormGradientJ = np.zeros((EPOCHS, N_AGENTS))

# Initialization for Accuracy
successes = np.zeros((N_AGENTS))
errors = np.zeros((N_AGENTS))
percentage_of_success = np.zeros((N_AGENTS))

###############################################################################
# Initialization of Gradient Tracking Algorithm
for agent in range(N_AGENTS):
    batch_grad[agent] = 0
    for idx in range(BATCH_SIZE):
        
        # Forward pass
        xx = forward_pass(images[agent, idx], uu[agent])

        # Loss evaluation
        out_grad = np.zeros((D_NEURONS)) # Initialize output gradient to 0 (for output regression)out_grad
        loss, out_grad[0] = cost_fn(labels[agent, idx], xx[-1, 0])

        # Backward pass
        _, grad = backward_pass(xx, uu[agent], out_grad) # out_grad = llambdaT

        # Gradient accumulation
        batch_grad[agent] += grad / BATCH_SIZE
        
    ss[agent] = batch_grad[agent]
        
###############################################################################
# Training
for epoch in range(EPOCHS):
    if epoch % 10 == 0 and epoch != 0:
        print(f'Cost at k={epoch:d} is {np.mean(J[epoch-1]):.4f}')

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
                xx = forward_pass(images[agent, idx], uu_plus[agent])
                prediction[agent,idx] = xx[-1, 0] # prediction <= value of the first neuron in the last layer

                # Loss evalutation
                out_grad = np.zeros((D_NEURONS)) # Initialize output gradient to 0 (for output regression)
                loss, out_grad[0] = cost_fn(labels[agent, idx], prediction[agent,idx])

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
            

uu_mean = np.mean(uu, axis=0)
for agent in range(N_AGENTS):
    print(f'The Agent {agent} has mean error = {np.linalg.norm(uu_mean - uu[agent])}')


print('\n\nTRAINING SET\n')
for agent in range(N_AGENTS):
    for batch_el in range(BATCH_SIZE):
        idx = ((N_BATCH-1)*BATCH_SIZE) + batch_el
        print(f"[Agent {agent}] Label for Image {idx} was {labels[agent,idx]} but is classified as :{prediction[agent,idx]:.4f}")
    print()


# ###############################################################################
# Accuracy computation
for agent in range(N_AGENTS):
    for img in range(SAMPLES_PER_AGENT):
        success, error = accuracy(prediction[agent,img],labels[agent,img])
        successes[agent] += success
        errors[agent] += error

    percentage_of_success[agent] = (successes[agent]/(SAMPLES_PER_AGENT))*100
    print('\nAGENT: ', agent)
    print("Correctly classified point: ", successes[agent])
    print("Wrong classified point: ", errors[agent])
    print(f"Percentage of Success: {percentage_of_success[agent]:.4f}" )   

                
###############################################################################
# PLOT
###############################################################################
plt.figure('Cost function')
plt.plot(range(EPOCHS),np.sum(J, axis=-1)/N_AGENTS, label='Total Normalized Cost Evolution', linewidth = 3)
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
     plt.plot(range(EPOCHS), NormGradientJ[:, agent], linestyle = ':')
plt.xlabel(r'Epochs')
plt.legend()
plt.title('norm_gradient_J')
plt.grid()

plt.show()