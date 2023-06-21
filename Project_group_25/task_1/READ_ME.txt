## Folder organization

The /lib/ folder contains the following files:
- config.py: to set all dataframe, network and training settings
- data_load.py: to load, balance and preprocess the mnist dataset 
- graph.py: contains all the connection settings (graph type, metropolis Hastings weights, ...)
- network_dynamics.py: contains all the functions used in the network training and the accuracy
- plot.py: contains functions for saving plots during training

The main file is "NN_Distributed_Gradient_Tracking.py" that executes the distributed training of the network and the test.


# How to run a simulation

Open the /lib/ folder and set in the config.py file the setup for the simulation

Run from VSCode "NN_Distributed_Gradient_Tracking.py"




