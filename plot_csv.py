
# 11/05/2022
# IN-LP
#
import numpy as np
import matplotlib.pyplot as plt
import signal
import os
signal.signal(signal.SIGINT, signal.SIG_DFL)

N_obstacle = 8

_, _, files = next(os.walk("./task_2.4/_csv_file"))
NN = len(files)

xx_csv = {}
Tlist = []

for ii in range(NN):
    xx_csv[ii] = np.genfromtxt("task_2.4/_csv_file/agent_{}.csv".format(ii), delimiter=',').T
    Tlist.append(xx_csv[ii].shape[1])

n_x = xx_csv[ii].shape[0]
print(f'Number of dimensions = {n_x}')
Tmax = min(Tlist)

xx = np.zeros((NN*n_x,Tmax))

for ii in range(NN):
    for jj in range(n_x):
        index_ii = ii*n_x+jj
        xx[index_ii,:] = xx_csv[ii][jj][:Tmax] # useful to remove last samples

plt.figure()
for x in xx[0:n_x*(NN-N_obstacle)]:
    plt.plot(range(Tmax), x)  

block_var = False if n_x < 3 else True
plt.show(block=block_var)



if 1 and n_x == 2: # animation 
    plt.figure()
    plt.title('Animation XY')
    dt = 3 # sub-sampling of the plot horizon
    for tt in range(0,Tmax,dt):
        xx_tt = xx[:,tt].T
        for ii in range((NN-N_obstacle)):
            index_ii =  ii*n_x + np.arange(n_x)
            xx_ii = xx_tt[index_ii]
            if ii%2 == 1: # Leaders are blue stars
                plt.plot(xx_ii[0],xx_ii[1], marker='*', markersize=12, fillstyle='full', color = 'tab:blue')
            else: # Followers are red circles
                plt.plot(xx_ii[0],xx_ii[1], marker='o', markersize=10, fillstyle='full', color = 'tab:red')

        if N_obstacle:
            for ii in range(N_obstacle):
                ii = (NN-N_obstacle) + ii
                index_ii =  ii*n_x + np.arange(n_x)
                xx_ii = xx_tt[index_ii]
                plt.plot(xx_ii[0],xx_ii[1], marker='s', markersize=8, fillstyle='full', color = 'tab:green')


        axes_lim = (np.min(xx)-1,np.max(xx)+1)
        plt.xlim(axes_lim); plt.ylim(axes_lim)
        # plt.plot(xx[0:n_x*NN:n_x,:].T,xx[1:n_x*NN:n_x,:].T) #Dovresti printare fino a NN-(numero di obstacle)

        plt.plot(xx[0:n_x*(NN-N_obstacle):n_x,:].T,xx[1:n_x*(NN-N_obstacle):n_x,:].T, color= 'tab:gray') #Dovresti printare fino a NN-(numero di obstacle)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.grid()
        plt.axis('equal')
        
        plt.show(block=False)
        plt.pause(0.001)
        if tt < Tmax - dt - 1:
            plt.clf()
    plt.show()
