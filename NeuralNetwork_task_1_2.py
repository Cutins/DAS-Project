'''
20 may 2023
@ Giulia Cutini, Cenerini Simone, Riccardo Paolini

Multi-sample Neural-Network (Centralized Training)
'''
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow.keras as ks
import pandas as pd

###############################################################################
# Set seed for reproducibility
SEED = 25
np.random.seed(SEED)

###############################################################################
# DataFrame Settings
TARGET = 3
SIZE = (4,4)
SAMPLES = 50 # Put even number

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

images = np.array(df_train_balanced['image'].tolist())   # [SAMPLES, image_size]
labels = np.array(df_train_balanced['label'].tolist())  # [SAMPLES]

image_size = images.shape[-1]

print(f'Positive samples {np.sum(labels == 1)}')
print(f'Negative samples {np.sum(labels == 0)}')

###############################################################################
# Network setting
T_LAYERS = 3        # Number of layers
D_NEURONS = image_size      # Number of neurons for each layer
ActivationFunct = "Sigmoid" # {"Sigmoid", "ReLu", "HyTan"}
CostFunct = "BinaryCrossEntropy"     # {"Quadratic", "BinaryCrossEntropy"}

###############################################################################

# Cost Function
def cost_fn(Y,xT0):
    '''
    Y [Scalar]
    XT0 [Scalar]
    '''

    if CostFunct == "BinaryCrossEntropy":
            J = -(Y*np.log(xT0) + 1e-10)-((1-Y)*(np.log(1-xT0)) + 1e-10)
            dJ = -Y/(xT0 + 1e-10) + (1-Y)/(1-xT0 + 1e-10)


    if CostFunct == "Quadratic":
            J = (xT0 - Y)*(xT0 - Y)
            dJ = 2*(xT0 - Y)

         
    # if mask is not None:
    #         dJ = dJ*mask

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
EPOCHS = 1000
STEP_SIZE = 1e-1
BATCH_SIZE = 8 # Dimension of the minibatch set
N_BATCH = int(np.ceil(SAMPLES/BATCH_SIZE))

# Network Variables
xx = np.zeros((BATCH_SIZE, T_LAYERS, D_NEURONS))
uu = np.random.randn(T_LAYERS-1, D_NEURONS, D_NEURONS+1)*1e-2
prediction = np.zeros((SAMPLES))

J = np.zeros(EPOCHS) # Cost function
NormGradientJ = np.zeros(EPOCHS)

# Initialization for Accuracy
successes = 0
errors = 0

for epoch in range(EPOCHS):
    if epoch % 5 == 0 and epoch != 0:
        print(f'Cost at k={epoch:d} is {J[epoch-1]:.4f}')

    for batch_num in range(N_BATCH):
        batch_grad = 0

        for batch_el in range(BATCH_SIZE):
            idx = (batch_num*BATCH_SIZE) + batch_el
            
            # Skip if SAMPLES are finished (last minibatch)
            if idx >= SAMPLES:
                break

            xx[batch_el] = forward_pass(images[idx], uu)
            prediction[idx] =  xx[batch_el, -1, 0]
            out_grad = np.zeros((D_NEURONS)) # Initialize output gradient to 0 (for output regression)
            loss, out_grad[0] = cost_fn(labels[idx], prediction[idx])
            _, grad = backward_pass(xx[batch_el], uu, out_grad) # out_grad = llambdaT

            J[epoch] += loss / SAMPLES
            batch_grad += grad / BATCH_SIZE

        uu = uu - (STEP_SIZE * batch_grad)
        NormGradientJ[epoch] += np.linalg.norm(batch_grad) / N_BATCH

for img in range(SAMPLES):
    print(f"Label for Image {img} was {labels[img]} but is classified as:", prediction[img])


###############################################################################
# Accuracy computation
for img in range(SAMPLES):
    success, error = accuracy(prediction[img],labels[img])
    successes += success
    errors += error

percentage_of_success = (successes/SAMPLES)*100
print("Correctly classified point: ", successes)
print("Wrong classified point: ", errors)
print("Percentage of Success: ", percentage_of_success)


###############################################################################
# PLOT
###############################################################################
plt.figure('Cost function')
plt.plot(range(EPOCHS),J)
plt.title('J')
plt.grid()

plt.figure('Norm of Cost function')
plt.semilogy(range(EPOCHS), NormGradientJ)
plt.title('norm_gradient_J')
plt.grid()

plt.show()
                
