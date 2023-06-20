'''
20 may 2023
@ Giulia Cutini, Cenerini Simone, Riccardo Paolini

Multi-sample Neural-Network (Distributed Training)
'''

import numpy as np
import matplotlib.pyplot as plt

import os
import pickle
from copy import deepcopy

from lib.graph import *
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
weight_val      = np.zeros((EPOCHS, N_AGENTS))
weights_mag     = np.zeros((EPOCHS, N_AGENTS))
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

for epoch in range(EPOCHS):

    # Early stopping if performance are not improving
    if epoch >= 200 and np.all([np.mean(J[e]) > np.mean(J[epoch-11]) for e in range(epoch-10, epoch)]):
        break

    if epoch % 1 == 0 and epoch != 0:
        print(f'[k={epoch:d}] Cost is {np.mean(J[epoch-1]):.4f} and Grandient is {np.mean(NormGradientJ[epoch-1]):.4f}')

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
        weight_val[epoch][agent] = uu[agent][-1][0, 1]
        weights_mag[epoch][agent] = np.sum([np.abs(uu[agent][layer]).sum() / uu[agent][layer].size for layer in range(n_layers-1)])
        ss_mag[epoch][agent] = np.sum([np.abs(ss[0][agent][layer]).sum() / ss[0][agent][layer].size for layer in range(n_layers-1)])



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
    weights_file = f'Grad_Track-weights_{SIZE[0]}x{SIZE[1]}_E{EPOCHS}_S{SAMPLES}_B{BATCH_SIZE}.pkl'
    weights_path = os.path.join(os.getcwd(), 'task_1/weights', weights_file)
    with open(weights_path, 'wb') as f:
        pickle.dump(uu[0], f)



################################ PLOTS ###############################

#########   J and Grad_J    #########
plt.figure('Cost function')
plt.plot(range(EPOCHS), np.mean(J, axis=1), label='Mean of the Cost', linewidth=2)
for agent in range(N_AGENTS):
     plt.plot(range(EPOCHS), J[:, agent], label =f'Cost of agent {agent}', linewidth=0.5)
plt.xlabel('Epochs')
plt.ylabel(r"$\frac{1}{N} \sum_{i=1}^N J_i(u_i^k)$")
plt.title("Evolution of the cost")
plt.legend()
plt.grid()
# Salvataggio del grafico come file immagine
plot_path = os.path.join(os.getcwd(), 'task_1', 'Plots', PLOT_FOLDER, 'J.png')
plt.savefig(plot_path)

plt.figure('Gradient of Cost function')
plt.semilogy(range(EPOCHS), np.mean(NormGradientJ, axis=-1), label='Mean of the gradient ', linewidth=2)
for agent in range(N_AGENTS):
    plt.semilogy(range(EPOCHS), NormGradientJ[:, agent], label =f'Gradient of agent {agent}', linewidth=0.5)
plt.xlabel('Epochs')
plt.ylabel(r"$\frac{1}{N} \sum_{i=1}^N \nabla{J_i(u_i^k)}$")
plt.title('Evolution of the Gradient of J')
plt.legend()
plt.grid()
# Salvataggio del grafico come file immagine
plot_path = os.path.join(os.getcwd(), 'task_1', 'Plots', PLOT_FOLDER, 'norm_gradient_J.png')
plt.savefig(plot_path)

#########   Weights    #########
# Computes the mean error over uu
plt.figure('Weights (uu) magnitude difference')
weights_mag_mean = np.mean(weights_mag, axis=-1)
for agent in range(N_AGENTS):
    plt.semilogy(np.abs(weights_mag[:, agent] - weights_mag_mean), label =f'Agent {agent}', linewidth=0.5)
plt.xlabel('Epochs')
plt.ylabel(r"$||u_i^k|| - \frac{1}{N} \sum_{i=1}^N ||u_i^k||$")
plt.title('Weights magnitude - Weights mean magnitude')
plt.legend()
plt.grid()
# Salvataggio del grafico come file immagine
plot_path = os.path.join(os.getcwd(), 'task_1', 'Plots', PLOT_FOLDER, 'Weights_difference.png')
plt.savefig(plot_path)

plt.figure('Single weight')
weight_val_mean = np.mean(weight_val, axis=-1)
for agent in range(N_AGENTS):
    plt.semilogy(np.abs(weight_val[:, agent] - weight_val_mean), label =f'Agent {agent}', linewidth=0.5)
plt.xlabel('Epochs')
plt.title('Weight per agent - Weight mean value across agents')
plt.legend()
plt.grid()
# Salvataggio del grafico come file immagine
plot_path = os.path.join(os.getcwd(), 'task_1', 'Plots', PLOT_FOLDER, 'Single_weight.png')
plt.savefig(plot_path)

plt.figure('Weights (uu) magnitude')
for agent in range(N_AGENTS):
    plt.plot(weights_mag[:, agent], label =rf"$||u_{agent}||$", linewidth=0.5)
plt.xlabel('Epochs')
plt.ylabel(r"$||u_i^k||$")
plt.title('Weights magnitude evolution')
plt.legend()
plt.grid()
# Salvataggio del grafico come file immagine
plot_path = os.path.join(os.getcwd(), 'task_1', 'Plots', PLOT_FOLDER, 'Weights_evolution.png')
plt.savefig(plot_path)

#########   Local estimate    #########
plt.figure('S evolution')
plt.semilogy(np.mean(ss_mag, axis=-1), label='S mean', linewidth=2)
for agent in range(N_AGENTS):
    plt.semilogy(ss_mag[:, agent], label =rf"$||s_{agent}||$", linewidth = 0.5)
plt.xlabel(r'Epochs')
plt.ylabel(r"$||s_i^k||$")
plt.title('Local estimate magnitude evolution')
plt.legend()
plt.grid()
# Salvataggio del grafico come file immagine
plot_path = os.path.join(os.getcwd(), 'task_1', 'Plots', PLOT_FOLDER, 'S_evolution.png')
plt.savefig(plot_path)


#########   Reduced plots    #########
# Computes the mean error over uu
plt.figure('Weights (uu) magnitude difference - 60 Epochs')
weights_mag_mean = np.mean(weights_mag, axis=-1)
for agent in range(N_AGENTS):
    plt.semilogy(np.abs(weights_mag[0:60, agent] - weights_mag_mean[0:60]), label =f'Agent {agent}', linewidth=0.5)
plt.xlabel('Epochs')
plt.ylabel(r"$||u_i^k|| - \frac{1}{N} \sum_{i=1}^N ||u_i^k||$")
plt.title('Weights magnitude - Weights mean magnitude')
plt.legend()
plt.grid()
# Salvataggio del grafico come file immagine
plot_path = os.path.join(os.getcwd(), 'task_1', 'Plots', PLOT_FOLDER, 'Weights_difference-60.png')
plt.savefig(plot_path)

plt.figure('Single weight - 60 Epochs')
weight_val_mean = np.mean(weight_val, axis=-1)
for agent in range(N_AGENTS):
    plt.semilogy(np.abs(weight_val[0:60, agent] - weight_val_mean[0:60]), label =f'Agent {agent}', linewidth=0.5)
plt.xlabel('Epochs')
plt.title('Weight per agent - Weight mean value across agents')
plt.legend()
plt.grid()
# Salvataggio del grafico come file immagine
plot_path = os.path.join(os.getcwd(), 'task_1', 'Plots', PLOT_FOLDER, 'Single_weight-20.png')
plt.savefig(plot_path)

plt.figure('Weights (uu) magnitude - 20 Epochs')
for agent in range(N_AGENTS):
    plt.plot(weights_mag[0:60, agent], label =rf"$||u_{agent}||$", linewidth=0.5)
plt.xlabel('Epochs')
plt.ylabel(r"$||u_i^k||$")
plt.title('Weights magnitude evolution')
plt.legend()
plt.grid()
# Salvataggio del grafico come file immagine
plot_path = os.path.join(os.getcwd(), 'task_1', 'Plots', PLOT_FOLDER, 'Weights_evolution-20.png')
plt.savefig(plot_path)

plt.figure('S evolution - 60 Epochs')
plt.semilogy(np.mean(ss_mag[0:60], axis=-1), label='S mean', linewidth=2)
for agent in range(N_AGENTS):
    plt.semilogy(ss_mag[0:60, agent], label =rf"$||s_{agent}||$", linewidth = 0.5)
plt.xlabel(r'Epochs')
plt.ylabel(r"$||s_i^k||$")
plt.title('Local estimate magnitude evolution')
plt.legend()
plt.grid()
# Salvataggio del grafico come file immagine
plot_path = os.path.join(os.getcwd(), 'task_1', 'Plots', PLOT_FOLDER, 'S_evolution-60.png')
plt.savefig(plot_path)

plt.show()