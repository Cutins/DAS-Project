# 20 may 2023
# Giulia Cutini

# Multi-sample Neural-Network (Centralized Training)

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

###############################################################################
EPOCHS = 4000
STEPSIZE = 1e-1

D_NEURONS = 16 # Number of neurons for each layer
T_LAYERS = 3 # Number of layers
I_BATCH = 5 # Dimension of the batch set

ActivationFunct = "HyTan" # {"Sigmoid", "ReLu", HyTan}
CostFunct = "Quadratic" # {"Quadratic", "BinaryCrossEntropy"}

###############################################################################
# Cost Function
def cost_fucnt(Y,xT, mask=None):

    if mask is not None:
            Y = Y*mask
            xT = xT*mask

    if CostFunct == "BinaryCrossEntropy":
            J = Y*np.log(xT) + (1-Y)*np.log(1-xT)
            dJ = (xT - Y)/(xT*(1-Y))
    #print(xT - Y)
    if CostFunct == "Quadratic":
            J = (xT - Y).T@(xT - Y)
            dJ = 2*(xT - Y)
    #print(dJ)
    return J, dJ

# Activation Function
def activation_funct(xl):

    if ActivationFunct == "ReLu": # Rectified Linear Unit
        out = max(0,xl) # xl * (xl>0)

    if ActivationFunct == "HyTan": # Hyperbolic tangent
        out = np.tanh(xl)

    if ActivationFunct == "Sigmoid": #Sigmoid function
        out = 1/(1+np.exp(-xl))

    return out

# Derivative of Activation Function
def activation_funct_derivative(xl):

    if ActivationFunct == "ReLu": # Rectified Linear Unit
        if xl>0:
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
    df_dx = np.zeros((D_NEURONS,D_NEURONS))
    df_du = np.zeros(((D_NEURONS+1)*D_NEURONS,D_NEURONS))

    dim = np.tile([D_NEURONS+1],D_NEURONS)
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
CurrentDir = os.path.dirname(os.path.abspath(__file__))

Name = ["000005.jpg", "000007.jpg", "000009.jpg", "000012.jpg", "000022.jpg"]
#Name = ["000005.jpg", "000007.jpg"]

INPUTS = np.zeros((I_BATCH, D_NEURONS)) # 3 Flattened Images with 16 pixels
LABEL = np.ones((I_BATCH, D_NEURONS))
if ActivationFunct == "HyTan": # Hyperbolic tangent
    LABEL[0,:] = -1 #Le immagini sono tutte corde, la prima è un martello
if ActivationFunct == "Sigmoid": #Sigmoid function
    LABEL[0,:] = 0 #Le immagini sono tutte corde, la prima è un martello

for i in range(len(Name)):
    Path = os.path.join(CurrentDir, "FlattenInput", "Flatten" + Name[i])
    Input = np.array(Image.open(Path))
    InputNormalized = Input/255.
    INPUTS[i,:] = InputNormalized.reshape(16,)
    # LABEL[i] = np.mean(INPUTS[i,:]) # Label is mean value of output layer

############## Plot of images
# fig, ax = plt.subplots(1, len(Name), figsize=(20,10))

xx = np.zeros((I_BATCH, T_LAYERS, D_NEURONS))

# for j in range(len(Name)):
#     ax[j].imshow(INPUTS[j,:].reshape(16,1), cmap = "gray", vmin = 0, vmax = 1)
    
# plt.show()
##############

J = np.zeros(EPOCHS) # Cost function
NormGradientJ = np.zeros(EPOCHS)

# uu = np.zeros(shape=(T_LAYERS-1,D_NEURONS,D_NEURONS+1)) 
uu =  np.random.randn(T_LAYERS-1, D_NEURONS, D_NEURONS+1)*1e-1

for i in range(I_BATCH):
    xx[i] = forward_pass(INPUTS[i],uu)

mask = np.zeros(D_NEURONS)
mask[0] = 1

# GO!
for k in range(EPOCHS):
    if k%1000==1:
        print(f'Cost at k={k:d} is {J[k-1]:.4f}')

    delta_u = 0
    for i in range(I_BATCH):

        _, llambdaT = cost_fucnt(LABEL[i],xx[i,-1,:],mask)
        _, Grad = backward_pass(xx[i],uu,llambdaT)

        delta_u += Grad/I_BATCH
        
    NormGradientJ[k] = np.linalg.norm(delta_u)
        
    uu = uu - STEPSIZE*delta_u
    
    for i in range(I_BATCH):
        xx[i] = forward_pass(INPUTS[i],uu)
        # print(cost_fucnt(LABEL[i],xx[i,-1,:],mask)[0].shape)
        J[k] += cost_fucnt(LABEL[i],xx[i,-1,:],mask)[0]

    # print(f'Output at k={k:d} is \n {xx[-1,0]}')
    # print(f'Cost at k={k:d} is {J[k-1]:.4f}')

    if k == EPOCHS-1:
        for i in range(I_BATCH):
            print(f"Label for Image {i} was {LABEL[i]} but is classified as:", xx[i,-1, 0])

_, ax = plt.subplots()
ax.plot(range(EPOCHS),J)
plt.grid()

_, ax = plt.subplots()
ax.semilogy(range(EPOCHS), NormGradientJ)
plt.grid()
plt.show()
