import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


#####################################################################################
#  Generate Network Graph
def get_graph(graph_type, n_agents):

    if graph_type == "Cycle":
        graph = nx.path_graph(n_agents)
        graph.add_edge(n_agents-1,0)
        nx.draw_circular(graph, with_labels=True)

    if graph_type == "Path":
        graph = nx.path_graph(n_agents)
        nx.draw(graph)
        
    if graph_type == "Star":
        graph = nx.star_graph(n_agents-1)
        nx.draw(graph, with_labels=True)

    #plt.show()

    id_agent = np.identity(n_agents, dtype=int)

    while 1:
        adj = nx.adjacency_matrix(graph)
        adj = adj.toarray()	

        test = np.linalg.matrix_power((id_agent+adj), n_agents)
        
        if np.all(test>0):
            print("the graph is connected\n")
            break 
        else:
            print("the graph is NOT connected\n")
            quit()

    print(adj)
    return graph, adj


#####################################################################################
# Metropolis Hastings
def get_weight_matrix(adj):  
    n_agents = adj.shape[0]
    degree = np.sum(adj, axis=0)
    WW = np.zeros((n_agents, n_agents))

    for agent in range(n_agents):
        Nii = np.nonzero(adj[agent])[0]
    
        for neigh in Nii:
            WW[agent,neigh] = 1 / (1 + np.max([degree[agent], degree[neigh]]))

        WW[agent,agent] = 1 - np.sum(WW[agent, :])

    print('Row Stochasticity {}'.format(np.sum(WW, axis=1)))
    print('Col Stochasticity {}'.format(np.sum(WW, axis=0)))

    return WW
