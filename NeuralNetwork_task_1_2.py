# 20 may 2023
# Giulia Cutini

# Multi-sample Neural-Network (Centralized Training)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import networkx as nx
import pickle

###############################################################################
# Set seed for reproducibility
SEED = 25
np.random.seed(SEED)

# Load DataFrame
file = open('dataset.pkl', 'rb')
df = pickle.load(file)
image_size = df['image'][0].size
file.close()

# Network setting
T_LAYERS = 3        # Number of layers
D_NEURONS = image_size      # Number of neurons for each layer
ActivationFunct = "Sigmoid" # {"Sigmoid", "ReLu", "HyTan"}
CostFunct = "Quadratic"     # {"Quadratic", "BinaryCrossEntropy"}

###############################################################################
# Create Train & Test split
df_train, df_test = train_test_split(df, test_size=0.1, random_state=SEED, shuffle=True)
df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

# Divide data in different sets - one for each agent
samples = len(df_train)

images = np.array(df_train['image'][:samples]) # [samples, image_size]
labels = np.array(df_train['label'][:samples]) # [samples]

###############################################################################

# Cost Function
def cost_fn(Y,xT, mask=None):
    xT0 = xT[0]

    if CostFunct == "BinaryCrossEntropy":
            J = -(Y*np.log(xT0) + (1-Y)*(np.log(1-xT0)))
            # dJ = (Y/xT0) + (1-Y)/(1-xT0)
            dJ = (xT0-Y)/(xT0*(1-xT0)+1e-4) #Preso da cidice di Nicholas


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

###############################################################################
# MAIN
###############################################################################

# Training parameters
EPOCHS = 1000
STEP_SIZE = 1e-1
BATCH_SIZE = 32 # Dimension of the minibatch set
N_BATCH = np.ceil(samples/BATCH_SIZE)

# Network Variables
xx = np.zeros((BATCH_SIZE, T_LAYERS, D_NEURONS))
uu = np.random.randn(T_LAYERS-1, D_NEURONS, D_NEURONS+1)*1e-2

# Mask for output regression
mask = np.zeros(D_NEURONS)
mask[0] = 1

J = np.zeros(EPOCHS) # Cost function
NormGradientJ = np.zeros(EPOCHS)

for epoch in range(EPOCHS):
    if epoch % 10 == 0 and epoch != 0:
        print(f'Cost at k={epoch:d} is {J[epoch-1]:.4f}')

    for batch_num in range(N_BATCH):
        batch_grad = 0

        for batch_el in range(BATCH_SIZE):
            idx = (batch_num*BATCH_SIZE) + batch_el
            
            # Skip if samples are finished (last minibatch)
            if idx >= samples:
                break

            xx[batch_el] = forward_pass(images[idx], uu)
            pred = xx[batch_el, -1, :]
            loss, out_grad = cost_fn(labels[idx], pred, mask)
            _, grad = backward_pass(xx[batch_el], uu, out_grad) # out_grad = llambdaT

            J[epoch] += loss / samples
            batch_grad += grad / BATCH_SIZE

        uu = uu - (STEP_SIZE * batch_grad)
        NormGradientJ[epoch] += np.linalg.norm(batch_grad) / N_BATCH

                
