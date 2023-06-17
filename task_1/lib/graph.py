import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
np.random.seed(SEED)

from lib.config import *


#####################################################################################
#  Generate Network Graph
def get_graph():
    plt.figure('Graph')

    if GRAPH_TYPE == "Cycle":
        graph = nx.path_graph(N_AGENTS)
        graph.add_edge(N_AGENTS-1,0)
        nx.draw_circular(graph, with_labels=True)

    if GRAPH_TYPE == "Path":
        graph = nx.path_graph(N_AGENTS)
        nx.draw(graph)
        
    if GRAPH_TYPE == "Star":
        graph = nx.star_graph(N_AGENTS-1)
        nx.draw(graph, with_labels=True)

    # Salvataggio del grafico come file immagine
    plot_path = os.path.join(os.getcwd(), 'task_1', 'Plots', PLOT_FOLDER, 'Graph.png')
    plt.savefig(plot_path)

    return graph


#####################################################################################
# Adjacency Matrix
def get_adjacency(graph): 
    while 1:
        adjacency = nx.adjacency_matrix(graph)
        adjacency = adjacency.toarray()	

        identity = np.identity(N_AGENTS, dtype=int)
        test = np.linalg.matrix_power((identity+adjacency), N_AGENTS)
        
        if np.all(test>0):
            print("the graph is connected\n")
            break 
        else:
            print("the graph is NOT connected\n")
            quit()

    return adjacency


#####################################################################################
# Metropolis Hastings
def get_weights(adj):
    degree = np.sum(adj, axis=0)
    weights = np.zeros((N_AGENTS, N_AGENTS))

    for agent in range(N_AGENTS):
        Nii = np.nonzero(adj[agent])[0]
    
        for neigh in Nii:
            weights[agent,neigh] = 1 / (1 + np.max([degree[agent], degree[neigh]]))

        weights[agent,agent] = 1 - np.sum(weights[agent, :])

    print('Row Stochasticity {}'.format(np.sum(weights, axis=1)))
    print('Col Stochasticity {}'.format(np.sum(weights, axis=0)))

    return weights
