import matplotlib.pyplot as plt
import os

from lib.config import *

#########      J     #########
def plot_cost(J, epochs=None):
    if epochs is None:
        epochs = J.shape[0]

    # Create folder if does not exist
    folder_path = os.path.join(os.getcwd(), 'task_1', 'Plots', PLOT_FOLDER, f'epoch_{epochs}')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plt.figure('Cost function')
    plt.plot(np.mean(J[:epochs], axis=1), label=r"$\frac{1}{N} \sum_{i=1}^N J(u_i)$", linewidth=2)
    for agent in range(N_AGENTS):
        plt.plot(J[:epochs, agent], label =rf"$J(u_{agent})$", linewidth=0.5)
    
    plt.xlabel(r'Epochs $[k]$')
    plt.ylabel(r"$\frac{1}{N} \sum_{i=1}^N J(u_i^k)$")
    plt.title("Evolution of the cost function")
    plt.legend()
    plt.grid()
    plot_path = os.path.join(folder_path, 'J.png')
    plt.savefig(plot_path)

    plt.figure('Cost function - LOG')
    plt.semilogy(np.mean(J[:epochs], axis=1), label=r"$\frac{1}{N} \sum_{i=1}^N J(u_i)$", linewidth=2)
    for agent in range(N_AGENTS):
        plt.semilogy(J[:epochs, agent], label =rf"$J(u_{agent})$", linewidth=0.5)
    plt.xlabel(r'Epochs $[k]$')
    plt.ylabel(r"$\frac{1}{N} \sum_{i=1}^N J(u_i^k)$")
    plt.title("Evolution of the cost function")
    plt.legend()
    plt.grid()
    plot_path = os.path.join(folder_path, 'J-log.png')
    plt.savefig(plot_path)



#########   Grad_J    #########
def plot_cost_grad(NormGradientJ, epochs=None):
    if epochs is None:
        epochs = NormGradientJ.shape[0]

    # Create folder if does not exist
    folder_path = os.path.join(os.getcwd(), 'task_1', 'Plots', PLOT_FOLDER, f'epoch_{epochs}')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plt.figure('Gradient of Cost function')
    plt.semilogy(np.mean(NormGradientJ[:epochs], axis=-1), label=r"$\frac{1}{N} \sum_{i=1}^N \nabla{J(u_i)}$", linewidth=2)
    for agent in range(N_AGENTS):
        plt.semilogy(NormGradientJ[:epochs, agent], label =fr"$\nabla J(u_{agent})$", linewidth=0.5)
    plt.xlabel(r'Epochs $[k]$')
    plt.ylabel(r"$\frac{1}{N} \sum_{i=1}^N \nabla{J(u_i^k)}$")
    plt.title('Evolution of the Gradient of the Cost function')
    plt.legend()
    plt.grid()
    plot_path = os.path.join(folder_path, 'norm_gradient_J.png')
    plt.savefig(plot_path)


#########   Single Weight   #########
def plot_weights_val(weight_val, epochs=None, step=1):
    if epochs is None:
        epochs = weight_val.shape[0]
    iters = epochs * N_BATCH

    # Create folder if does not exist
    folder_path = os.path.join(os.getcwd(), 'task_1', 'Plots', PLOT_FOLDER, f'epoch_{epochs}')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plt.figure(f'Single weight difference (step: {step})')
    weight_val_mean = np.mean(weight_val[:iters:step], axis=-1)
    for agent in range(N_AGENTS):
        plt.plot(range(0, iters, step), weight_val[:iters:step, agent] - weight_val_mean, label =f'Agent {agent}', linewidth=0.5)
    plt.xlabel(r'Updates $[h]$')
    plt.ylabel(r"$\hat{u}_i^h - \frac{1}{N} \sum_{i=1}^N \hat{u}_i^h$")
    plt.title('Weight per agent - Weight mean value across agents')
    plt.legend()
    plt.grid()
    plot_path = os.path.join(folder_path, f'Single_weight_difference_step_{step}.png')
    plt.savefig(plot_path)


    plt.figure(f'Single weight difference - LOG (step: {step})')
    weight_val_mean = np.mean(weight_val[:iters:step], axis=-1)
    for agent in range(N_AGENTS):
        plt.semilogy(range(0, iters, step), np.abs(weight_val[:iters:step, agent] - weight_val_mean), label =f'Agent {agent}', linewidth=0.5)
    plt.xlabel(r'Updates $[h]$')
    plt.ylabel(r"$\hat{u}_i^h - \frac{1}{N} \sum_{i=1}^N \hat{u}_i^h$")
    plt.title('Weight per agent - Weight mean value across agents')
    plt.legend()
    plt.grid()
    plot_path = os.path.join(folder_path, f'Single_weight_difference-log_step_{step}.png')
    plt.savefig(plot_path)


    plt.figure(f'Single weight(step: {step})')
    for agent in range(N_AGENTS):
        plt.plot(range(0, iters, step), weight_val[:iters:step, agent], label =rf"$||\hat u_{agent}||$", linewidth=0.5)
    plt.xlabel(r'Updates $[h]$')
    plt.ylabel(r"$\hat{u}_i^h$")
    plt.title('Single weight evolution')
    plt.legend()
    plt.grid()
    plot_path = os.path.join(folder_path, f'Single_weight_step_{step}.png')
    plt.savefig(plot_path)


#########   Weights   #########
def plot_weights_mag(weights_mag, epochs=None, step=1):
    if epochs is None:
        epochs = weights_mag.shape[0]
    iters = epochs*N_BATCH

    # Create folder if does not exist
    folder_path = os.path.join(os.getcwd(), 'task_1', 'Plots', PLOT_FOLDER, f'epoch_{epochs}')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plt.figure(f'Weights (uu) magnitude difference (step: {step})')
    weights_mag_mean = np.mean(weights_mag[:iters:step], axis=-1)
    for agent in range(N_AGENTS):
        plt.plot(range(0, iters, step), weights_mag[:iters:step, agent] - weights_mag_mean, label =f'Agent {agent}', linewidth=0.5)
    plt.xlabel(r'Updates $[h]$')
    plt.ylabel(r"$||u_i^h|| - \frac{1}{N} \sum_{i=1}^N ||u_i^h||$")
    plt.title('Weights magnitude - Weights mean magnitude')
    plt.legend()
    plt.grid()
    plot_path = os.path.join(folder_path, f'Weights_difference_step_{step}.png')
    plt.savefig(plot_path)


    plt.figure(f'Weights (uu) magnitude difference - LOG (step: {step})')
    weights_mag_mean = np.mean(weights_mag[:iters:step], axis=-1)
    for agent in range(N_AGENTS):
        plt.semilogy(range(0, iters, step), np.abs(weights_mag[:iters:step, agent] - weights_mag_mean), label =f'Agent {agent}', linewidth=0.5)
    plt.xlabel(r'Updates $[h]$')
    plt.ylabel(r"$||u_i^h|| - \frac{1}{N} \sum_{i=1}^N ||u_i^h||$")
    plt.title('Weights magnitude - Weights mean magnitude')
    plt.legend()
    plt.grid()
    plot_path = os.path.join(folder_path, f'Weights_difference-log_step_{step}.png')
    plt.savefig(plot_path)


    plt.figure(f'Weights (uu) magnitude (step: {step})')
    for agent in range(N_AGENTS):
        plt.plot(range(0, iters, step), weights_mag[:iters:step, agent], label =rf"$||u_{agent}||$", linewidth=0.5)
    plt.xlabel(r'Updates $[h]$')
    plt.ylabel(r"$||u_i^h||$")
    plt.title('Weights magnitude evolution')
    plt.legend()
    plt.grid()
    plot_path = os.path.join(folder_path, f'Weights_evolution_step_{step}.png')
    plt.savefig(plot_path)


def plot_ss_mag(ss_mag, epochs=None):
    if epochs is None:
        epochs = ss_mag.shape[0]

    # Create folder if does not exist
    folder_path = os.path.join(os.getcwd(), 'task_1', 'Plots', PLOT_FOLDER, f'epoch_{epochs}')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    #########   Local estimate    #########
    plt.figure('S evolution')
    plt.semilogy(np.mean(ss_mag[:epochs], axis=-1), label=r"$\frac{1}{N} \sum_{i=1}^N ||s_i||$", linewidth=2)
    for agent in range(N_AGENTS):
        plt.semilogy(ss_mag[:epochs, agent], label =rf"$||s_{agent}||$", linewidth = 0.5)
    plt.xlabel(r'Epochs $[k]$')
    plt.ylabel(r"$\frac{1}{N} \sum_{i=1}^N ||s_i^k||$")
    plt.title('Local estimate magnitude evolution')
    plt.legend()
    plt.grid()
    plot_path = os.path.join(folder_path, 'S_evolution.png')
    plt.savefig(plot_path)