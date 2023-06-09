#
# Formation Control Algorithm
# Lorenzo Pichierri
# Bologna, 18/03/2022
#

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import networkx as nx


def waves(amp, omega, phi, t, n_x, n_agents):
    """
    This function generates a sinusoidal input trajectory

    Input:
        - amp, omega, phi = sine parameter u = amp*sin(omega*t + phi)
        - n_agents = number of agents
        - n_x = agent dimension
        - t = time variable

    Output:
        - u = input trajectory

    """

    u = [amp*np.sin(omega*t+phi) for _ in range(n_x*n_agents)]
    return np.array(u)

def animation(xx, NN, n_x, n_leaders, horizon, dt, animate=True):
    """
    This function generates the containment animation

    """

    TT = np.size(horizon,0)
    # for tt in range(0,TT,int(TT/200)):
    for tt in range(0,TT,dt):
        xx_tt = xx[:,tt].T

        # Plot trajectories
        if tt>dt and tt<TT-1:
            plt.plot(xx[0:n_x*(NN-n_leaders):n_x,tt-dt:tt+1].T,xx[1:n_x*(NN-n_leaders):n_x,tt-dt:tt+1].T, linewidth = 2, color = 'tab:blue')
            plt.plot(xx[n_x*(NN-n_leaders):n_x*NN:n_x,tt-dt:tt+1].T,xx[n_x*(NN-n_leaders)+1:n_x*NN:n_x,tt-dt:tt+1].T, linewidth = 3, color = 'tab:red')

        # Plot convex hull
        leaders_pos = np.reshape(xx[n_x*(NN-n_leaders):n_x*NN,tt],(n_leaders,n_x))
        hull = ConvexHull(leaders_pos)
        plt.fill(leaders_pos[hull.vertices,0], leaders_pos[hull.vertices,1], 'darkred', alpha=0.3)
        vertices = np.hstack((hull.vertices,hull.vertices[0])) # add the firt in the last position to draw the last line
        plt.plot(leaders_pos[vertices,0], leaders_pos[vertices,1], linewidth = 2, color = 'darkred', alpha=0.7)


        # Plot agent position
        for ii in range(NN):
            index_ii =  ii*n_x + np.array(range(n_x))
            p_prev = xx_tt[index_ii]
            agent_color = 'blue' if ii < NN-n_leaders else 'red'
            plt.plot(p_prev[0],p_prev[1], marker='o', markersize=10, fillstyle='full', color = agent_color)
    
    
        x_lim = (np.min(leaders_pos[hull.vertices,0])-1,np.max(leaders_pos[hull.vertices,0])+1)
        y_lim = (np.min(leaders_pos[hull.vertices,1])-1,np.max(leaders_pos[hull.vertices,1])+1)
        # axes_lim = (0,0)
        # plt.axis('equal')
        plt.title("Agents position in $\mathbb{R}^2$")
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.show(block=False)
        plt.pause(0.1)

        plt.clf()