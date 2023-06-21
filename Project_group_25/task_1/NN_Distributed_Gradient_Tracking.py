'''
21 June 2023
@ Giulia Cutini, Cenerini Simone, Riccardo Paolini

Multi-sample Neural-Network (Distributed Training with Gradient Tracking)
'''

import numpy as np
import matplotlib.pyplot as plt

import os
import pickle
from copy import deepcopy

from lib.graph import *
from lib.plots import *
from lib.config import *
from lib.data_load import *
from lib.network_dynamics import *

# Set seed for reproducibility
np.random.seed(SEED)

folder_path = os.path.join(os.getcwd(), 'task_1', 'Plots', PLOT_FOLDER)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# DATA LOADING
images_train, labels_train, images_test, labels_test = load_data()
image_size = images_train.shape[-1]

print(f'Total positive samples {np.sum(labels_train == 1)}')
print(f'Total negative samples {np.sum(labels_train == 0)}')

# GRAPH CREATION
graph = get_graph()   

# ADJACENCY MATRIX
ADJ = get_adjacency(graph)                              

# METROPOLIS HASTINGS WEIGHTS
WW = get_weights(ADJ)                               


######################################################################
################################ MAIN ################################
######################################################################

n_layers = len(NETWORK)

# NN Weights
uu = [1e-1*np.random.randn(NETWORK[layer_idx+1], NETWORK[layer_idx]+1) for layer_idx in range(n_layers-1)]
uu = [deepcopy(uu) for _ in range(N_AGENTS)]                        
uu_plus = deepcopy(uu)  # shape -> (N_AGENTS, weights.shape)          

# NN Auxiliary variable
ss = [np.zeros((NETWORK[layer_idx+1], NETWORK[layer_idx]+1)) for layer_idx in range(n_layers-1)]
ss = [[deepcopy(ss) for _ in range(N_AGENTS)] for _ in range(N_BATCH)]  
ss_plus = deepcopy(ss)  # shape -> (N_BATCH, N_AGENTS, weights.shape)                                            

# NN Gradients
old_grad = [np.zeros((NETWORK[layer_idx+1], NETWORK[layer_idx]+1)) for layer_idx in range(n_layers-1)]                             
old_grad = [[deepcopy(old_grad) for _ in range(N_AGENTS)] for _ in range(N_BATCH)] # shape[N_BATCH, N_AGENTS, weights.shape]

'''
GRADIENT TRACKING THEORY
uu       -> x[k]      
uu_plus  -> x[k+1]

ss       -> s[k]      
ss_plus  -> s[k+1] 

old_grad -> ∇f(x[k])
grad     -> ∇f(x[k+1])
'''

# Save predictions, loss and gradients magnitude
prediction      = np.zeros((N_AGENTS, SAMPLES))
J               = np.zeros((EPOCHS, N_AGENTS))
NormGradientJ   = np.zeros((EPOCHS, N_AGENTS))

# Values to plot
weight_val      = np.zeros((EPOCHS*N_BATCH, N_AGENTS))
weights_mag     = np.zeros((EPOCHS*N_BATCH, N_AGENTS))
ss_mag          = np.zeros((EPOCHS, N_AGENTS))

# Initialization for evaluating accuracy
successes_train = np.zeros((N_AGENTS))
successes_test  = np.zeros((N_AGENTS))

errors_train    = np.zeros((N_AGENTS))
errors_test     = np.zeros((N_AGENTS))

accuracy_train  = np.zeros((N_AGENTS))
accuracy_test   = np.zeros((N_AGENTS))



############################ INITIALIZATION ###########################

# Initialization of Gradient Tracking Algorithm
for batch_num in range(N_BATCH):
    for agent in range(N_AGENTS):
        for batch_el in range(BATCH_SIZE):
            idx = (batch_num*BATCH_SIZE) + batch_el

            # Skip if samples are finished (last minibatch)
            if idx >= SAMPLES_PER_AGENT:
                break
            
            # Forward pass
            xx = forward_pass(images_train[agent, idx], uu[agent])

            # Loss evaluation
            loss, out_grad = cost_fn(labels_train[agent, idx], xx[-1])

            # Backward pass
            _, grad = backward_pass(xx, uu[agent], out_grad) # out_grad = llambdaT

            # Gradient accumulation
            for layer in range(n_layers-1):
                old_grad[batch_num][agent][layer] += grad[layer] / BATCH_SIZE

        for layer in range(n_layers-1):
            ss[batch_num][agent][layer] = old_grad[batch_num][agent][layer]



############################### TRAINING #############################

try:
    for epoch in range(EPOCHS):

        # Early stopping if performance are not improving
        if epoch >= 600 and np.all([np.mean(NormGradientJ[e]) > np.mean(NormGradientJ[epoch-501] - 1e-5) for e in range(epoch-500, epoch)]):
            plt.close('all')
            plot_cost(J, epoch-1)
            plot_cost_grad(NormGradientJ, epoch-1)
            if N_BATCH > 1:
                plot_weights_val(weight_val, epoch-1, step=N_BATCH)
                plot_weights_mag(weights_mag, epoch-1, step=N_BATCH)
            plot_weights_val(weight_val, epoch-1, step=1)
            plot_weights_mag(weights_mag, epoch-1, step=1)
            plot_ss_mag(ss_mag, epoch-1)
            break

        if epoch % 1 == 0 and epoch != 0:
            print(f'[k={epoch:d}] Cost is {np.mean(J[epoch-1]):.7f} and Grandient is {np.mean(NormGradientJ[epoch-1]):.7f}')

        for batch_num in range(N_BATCH):
            kk = epoch*N_BATCH+batch_num

            for agent in range(N_AGENTS):
                neighs = np.nonzero(ADJ[agent])[0]

                # Gradient Tracking Algorithm - Weights Update
                for layer in range(n_layers-1):
                    uu_plus[agent][layer] = (WW[agent, agent] * uu[agent][layer]) - (STEP_SIZE * ss[batch_num][agent][layer])
                    for neigh in neighs:
                        uu_plus[agent][layer] += WW[agent, neigh] * uu[neigh][layer]

                batch_grad = [np.zeros_like(ul) for ul in uu[agent]]
                for batch_el in range(BATCH_SIZE):
                    idx = (batch_num*BATCH_SIZE) + batch_el
                    
                    # Skip if samples are finished (last minibatch)
                    if idx >= SAMPLES_PER_AGENT:
                        break

                    # Forward pass
                    xx = forward_pass(images_train[agent, idx], uu_plus[agent])
                    prediction[agent, idx] = xx[-1] # prediction <= value of the first neuron in the last layer
                    
                    # Loss evalutation
                    loss, out_grad = cost_fn(labels_train[agent, idx], xx[-1])

                    # Backward pass
                    _, grad = backward_pass(xx, uu_plus[agent], out_grad) # out_grad = llambdaT

                    J[epoch, agent] += loss / SAMPLES_PER_AGENT
                    for layer in range(n_layers-1):
                        batch_grad[layer] += grad[layer] / BATCH_SIZE

                for layer in range(n_layers-1):
                    NormGradientJ[epoch, agent] += (np.abs(batch_grad[layer]).sum() / batch_grad[layer].size) / N_BATCH
                
                # Gradient Tracking Algorithm - SS Update
                for layer in range(n_layers-1):
                    ss_plus[batch_num][agent][layer] = (WW[agent, agent] * ss[batch_num][agent][layer]) + (batch_grad[layer] - old_grad[batch_num][agent][layer])
                    for neigh in neighs:
                        ss_plus[batch_num][agent][layer] += WW[agent, neigh] * ss[batch_num][neigh][layer]

                for layer in range(n_layers-1):
                    old_grad[batch_num][agent][layer] = batch_grad[layer]
            
            # Synch update
            for agent in range(N_AGENTS):
                for layer in range(n_layers-1):
                    uu[agent][layer] = uu_plus[agent][layer]
                    ss[batch_num][agent][layer] = ss_plus[batch_num][agent][layer]

            for agent in range(N_AGENTS):
                iteration = epoch*N_BATCH+batch_num
                weight_val[iteration][agent] = uu[agent][-1][0, 1]
                weights_mag[iteration][agent] = np.sum([np.abs(uu[agent][layer]).sum() / uu[agent][layer].size for layer in range(n_layers-1)])
                ss_mag[epoch][agent] = np.sum([np.abs(ss[0][agent][layer]).sum() / ss[0][agent][layer].size for layer in range(n_layers-1)])

        if (epoch + 1) % SAVE_STEP == 0 or epoch == (EPOCHS-1):
            plt.close('all')
            plot_cost(J, epoch)
            plot_cost_grad(NormGradientJ, epoch)
            if N_BATCH > 1:
                plot_weights_val(weight_val, epoch, step=N_BATCH)
                plot_weights_mag(weights_mag, epoch, step=N_BATCH)
            plot_weights_val(weight_val, epoch, step=1)
            plot_weights_mag(weights_mag, epoch, step=1)
            plot_ss_mag(ss_mag, epoch)

except KeyboardInterrupt:
    plt.close('all')
    plot_cost(J, epoch-1)
    plot_cost_grad(NormGradientJ, epoch-1)
    if N_BATCH > 1:
        plot_weights_val(weight_val, epoch-1, step=N_BATCH)
        plot_weights_mag(weights_mag, epoch-1, step=N_BATCH)
    plot_weights_val(weight_val, epoch-1, step=1)
    plot_weights_mag(weights_mag, epoch-1, step=1)
    plot_ss_mag(ss_mag, epoch-1)

########################## PRINT PREDICTIONS #########################

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
        print(f"[Agent {agent}] Label for the SAME Image {idx} was {labels_test[0, idx]} and is classified as {forward_pass(images_test[0, idx], uu[agent])[-1][0]:.4f}")
    print()



########################## COMPUTE ACCURACY ##########################

print('\n--------------TRAINING SCORES----------------')
for agent in range(N_AGENTS):
    for img in range(SAMPLES_PER_AGENT):
        success, error = accuracy(prediction[agent, img], labels_train[agent, img])
        successes_train[agent] += success 
        errors_train[agent] += error

    accuracy_train[agent] = (successes_train[agent] / (SAMPLES_PER_AGENT)) * 100
    print('\nAGENT: ', agent)
    print("Correctly classified point: ", successes_train[agent])
    print("Wrong classified point: ", errors_train[agent])
    print(f"Accuracy: {accuracy_train[agent]:.4f}" )  

print('\n------------------TEST SCORES-----------------')
for agent in range(N_AGENTS):
    for img in range(SAMPLES_PER_AGENT):
        output = forward_pass(images_test[0, img], uu[agent])
        success, error = accuracy(output[-1][0], labels_test[0, img])
        successes_test[agent] += success
        errors_test[agent] += error

    accuracy_test[agent] = (successes_test[agent] / (SAMPLES_PER_AGENT)) * 100
    print('\nAGENT: ', agent)
    print("Correctly classified point: ", successes_test[agent])
    print("Wrong classified point: ", errors_test[agent])
    print(f"Accuracy: {accuracy_test[agent]:.4f}" ) 



############################# SAVE WEIGHTS ###########################

# Save weights of agent 0 (theoretically at consensous)
if SAVE_WEIGHTS:
    weights_folder = os.path.join(os.getcwd(), 'task_1', 'weights')
    if not os.path.exists(weights_folder):
        os.makedirs(weights_folder)

    weights_path = os.path.join(weights_folder, f'Grad_Track-weights_{SIZE[0]}x{SIZE[1]}_E{EPOCHS}_S{SAMPLES}_B{BATCH_SIZE}.pkl')
    with open(weights_path, 'wb') as f:
        pickle.dump(uu[0], f)



################################ PLOTS ###############################
plt.show()