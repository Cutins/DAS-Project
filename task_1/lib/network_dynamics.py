import numpy as np
from lib.config import *
np.random.seed(SEED)


# Cost Function
def cost_fn(Y, xT):
    '''
    Y  [Scalar]
    XT [Scalar]
    '''

    if LOSS_TYPE == "BinaryCrossEntropy":
            J = -(Y*np.log(xT + 1e-5) )-((1-Y)*(np.log(1-xT + 1e-5)))
            dJ = -Y/(xT + 1e-5) + (1-Y)/(1-xT + 1e-5)


    if LOSS_TYPE == "Quadratic":
            J = (xT - Y)*(xT - Y)
            dJ = 2*(xT - Y)

    return J, dJ



# Activation Function
def activation_fn(xl):
    
    if ACTIVATION_TYPE == "ReLu":       # Rectified Linear Unit
        out = max(0, xl) 

    if ACTIVATION_TYPE == "HyTan":      # Hyperbolic tangent [-1, +1]
        out = np.tanh(xl)

    if ACTIVATION_TYPE == "Sigmoid":    # Sigmoid function [0, +1]
        out = 1 / (1 + np.exp(-xl))

    return out



# Derivative of Activation Function
def activation_fn_derivative(xl):

    if ACTIVATION_TYPE == "ReLu": # Rectified Linear Unit
        if xl > 0:
            out = 1
        else:
            out = 0

    if ACTIVATION_TYPE == "HyTan": # Hyperbolic tangent
        out = 1-(activation_fn(xl))**2

    if ACTIVATION_TYPE == "Sigmoid": # Sigmoid function
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