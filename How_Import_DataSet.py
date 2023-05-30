# 20 may 2023
# Giulia Cutini

# Multi-sample Neural-Network (Centralized Training)

import numpy as np
import cv2
import tensorflow.keras as ks
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from PIL import Image
import os
import pickle

###############################################################################
EPOCHS = 1000
STEPSIZE = 1e-1

D_NEURONS = 16 # Number of neurons for each layer
T_LAYERS = 3 # Number of layers
BATCHSIZE = 16 # Dimension of the minibatch set

N_AGENTS = 1
SEED = 25
np.random.seed(SEED)

ActivationFunct = "Sigmoid" # {"Sigmoid", "ReLu", "HyTan"}
CostFunct = "Quadratic" # {"Quadratic", "BinaryCrossEntropy"}

# Load data
(x_train, y_train), (x_test, y_test) = ks.datasets.mnist.load_data()

TARGET = 3
SIZE = (4,4)
N_AGENTS = 1
SAMPLES_PER_AGENT = 128

preprocess = lambda x: cv2.resize(x, SIZE).flatten() / 255.

x_train = [preprocess(x) for x in x_train]
x_test  = [preprocess(x) for x in x_test]

y_train = [1 if y == TARGET else 0 for y in y_train]
y_test  = [1 if y == TARGET else 0 for y in y_test]

x_train = np.array([x_train[SAMPLES_PER_AGENT*agent:SAMPLES_PER_AGENT*(agent+1)] for agent in range(N_AGENTS)]).squeeze()
y_train = np.array([y_train[SAMPLES_PER_AGENT*agent:SAMPLES_PER_AGENT*(agent+1)] for agent in range(N_AGENTS)]).squeeze()

print(x_train.shape)

N_BATCH = np.ceil(SAMPLES_PER_AGENT / BATCHSIZE)
###############################################################################
# Cost Function
def cost_fucnt(Y,xT, mask=None):

    if mask is not None:
            Y = Y*mask
            xT = xT*mask

    if CostFunct == "BinaryCrossEntropy":
            pass

    if CostFunct == "Quadratic":
            J = (xT - Y).T@(xT - Y)
            dJ = 2*(xT - Y)
            
    return J, dJ

# Activation Function
def activation_funct(xl):

    if ActivationFunct == "ReLu": # Rectified Linear Unit
        out = max(0, xl) # xl * (xl>0)

    if ActivationFunct == "HyTan": # Hyperbolic tangent [-1, +1]
        out = np.tanh(xl)

    if ActivationFunct == "Sigmoid": #Sigmoid function [0, +1]
        out = 1 / (1 + np.exp(-xl))

    return out

# Derivative of Activation Function
def activation_funct_derivative(xl):

    if ActivationFunct == "ReLu": # Rectified Linear Unit
        if xl > 0:
            out = 1
        else:
            out = 0

    if ActivationFunct == "HyTan": # Hyperbolic tangent
        out = 1-(activation_funct(xl))**2

    if ActivationFunct == "Sigmoid": # Sigmoid function
        out = activation_funct(xl)*(1-activation_funct(xl))

    return out

# Inference: xtp = f(xt,ut) (Activation Function for a whole layer)
def inference_dynamics(xt,ut):

    xtp = np.zeros(D_NEURONS)

    for l in range(D_NEURONS):
        temp = xt@ut[l,1:] + ut[l,0]
        xtp[l] = activation_funct(temp)

    return xtp
  
# Forward Propagation: (Inference dynamics for all the layers)
def forward_pass(x0,uu):
    xx = np.zeros((T_LAYERS,D_NEURONS))
    xx[0] = x0

    # Repeate the inference dynamics for all the layers
    for t in range(T_LAYERS-1):
        xx[t+1] = inference_dynamics(xx[t],uu[t])

    return xx
  
# Adjoint dynamics
def adjoint_dynamics(ltp, xt, ut):
    df_dx = np.zeros((D_NEURONS, D_NEURONS))
    df_du = np.zeros(((D_NEURONS+1)*D_NEURONS, D_NEURONS))

    dim = np.tile([D_NEURONS+1], D_NEURONS)
    cs_idx = np.append(0, np.cumsum(dim))
  
    for l in range(D_NEURONS):
        xl = xt@ut[l,1:] + ut[l,0]
        dSigma = activation_funct_derivative(xl)

        df_dx[:,l] = dSigma*ut[l,1:] # A matrix
        df_du[cs_idx[l]:cs_idx[l+1],l] = dSigma*np.hstack([1,xt])

    lt = df_dx@ltp # Adjoint equation
    delta_ut_vec = df_du@ltp
    delta_ut = np.reshape(delta_ut_vec,(D_NEURONS,D_NEURONS+1))

    return lt, delta_ut

# Backward Propagation: (Adjoint dynamics for all the layers)
def backward_pass(xx,uu,llambdaT):

    llambda = np.zeros((T_LAYERS,D_NEURONS))
    delta_u = np.zeros((T_LAYERS-1,D_NEURONS,D_NEURONS+1))

    llambda[-1] = llambdaT

    for t in reversed(range(T_LAYERS-1)):
        llambda[t], delta_u[t] = adjoint_dynamics(llambda[t+1],xx[t],uu[t])

    return llambda, delta_u



###############################################################################
# MAIN
###############################################################################


#### Load input images from processed folder
#CurrentDir = os.path.dirname(os.path.abspath(__file__))

#Name = ["000005.jpg", "000007.jpg", "000009.jpg", "000012.jpg", "000022.jpg"]

#INPUTS = np.zeros((BATCHSIZE, D_NEURONS)) # 3 Flattened Images with 16 pixels
#LABEL = np.ones((BATCHSIZE, D_NEURONS))

INPUTS = x_train # 3 Flattened Images with 16 pixels
#if ActivationFunct == 'HyTan':
#    df_train['label'][df_train['label'] == 0] = -1
LABEL = y_train
# print(INPUTS[1])

 #if ActivationFunct == "HyTan":
#    LABEL[0,:] = -1 #Le immagini sono tutte corde, la prima è un martello
#if ActivationFunct == "Sigmoid":
#    LABEL[0,:] = 0 #Le immagini sono tutte corde, la prima è un martello

#for i in range(len(Name)):
#    Path = os.path.join(CurrentDir, "FlattenInput", "Flatten" + Name[i])
#    Input = np.array(Image.open(Path))
#    InputNormalized = Input/255.
#    INPUTS[i,:] = InputNormalized.reshape(16,)

############## Plot of images
# fig, ax = plt.subplots(1, len(Name), figsize=(20,10))

xx = np.zeros((BATCHSIZE, T_LAYERS, D_NEURONS))

# for j in range(len(Name)):
#     ax[j].imshow(INPUTS[j,:].reshape(16,1), cmap = "gray", vmin = 0, vmax = 1)
    
# plt.show()
##############

J = np.zeros(EPOCHS) # Cost function
NormGradientJ = np.zeros(EPOCHS)

# uu = np.zeros(shape=(T_LAYERS-1,D_NEURONS,D_NEURONS+1)) 
uu =  np.random.randn(T_LAYERS-1, D_NEURONS, D_NEURONS+1)*1e-1

# for i in range(BATCHSIZE):
#     xx[i] = forward_pass(INPUTS[i],uu)

mask = np.zeros(D_NEURONS)
mask[0] = 1

# GO!
for k in range(EPOCHS):
    if k%10==0 and k!=0:
        print(f'Cost at k={k:d} is {J[k-1]:.4f}')

    for batch_n in range(N_BATCH):
        delta_u = 0
        for batch_sample in range(BATCHSIZE):
            idx = (batch_n*BATCHSIZE) + batch_sample
            if idx >= SAMPLES_PER_AGENT:
                continue
            
            xx[batch_sample] = forward_pass(INPUTS[idx], uu)

            loss, llambdaT = cost_fucnt(LABEL[idx], xx[batch_sample,-1,:], mask)
            J[k] += loss / SAMPLES_PER_AGENT
            _, Grad = backward_pass(xx[batch_sample], uu, llambdaT)

            delta_u += Grad / BATCHSIZE
        
        uu = uu - STEPSIZE*delta_u 
        NormGradientJ[k] += np.linalg.norm(delta_u) / N_BATCH
    
    if k == EPOCHS-1:
        for img in range(BATCHSIZE):
            print(f"Label for Image {img} was {LABEL[img]} but is classified as:", xx[img,-1, 0])





###############################################################################
# PLOT
###############################################################################
plt.figure('Cost function')
plt.plot(range(EPOCHS),J)
plt.title('J')

plt.figure('Norm of Cost function')
plt.semilogy(range(EPOCHS), NormGradientJ)
plt.title('norm_gradient_J')

plt.grid()
plt.show()


