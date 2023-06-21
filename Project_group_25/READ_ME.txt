## FOLDER ORGANIZATION

The /Project_group_25/ is organized as follows:
- report.group_25.pdf: report file of the project
- /report/ folder: contains the LaTeX file of the report and the /figs/ folder
- /task_1/: folder containing the code of first task
- /task_2/: folder containing the code of second task
- /videos/: folder that contains some of the more significant simulations



################################### FIRST TASK : NEURAL NETWORK #################################################
# Folder organization

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



##################################### SECOND TASK: FORMATION CONTROL ###############################################
## Folder organization

The folder contains the following subfolders:
- /task_2.2/: formation control with collision avoidance 
                in order to change the desidered configuration modify the CONFIG section in "formation_control.launch.py"
                e.g. change the number of agents (N) or get a 3d formation (_3d_formation)

- /task_2.3/: formation control with moving leaders
                to change the desired leaders' motion modify the TYPE_MOTION parameter in the CONFIG section of "formation_control.launch.py".

- /task_2.4/: formation control with obstacle avoidance
                to change the number of obstacles modify the N_OBSTACLES parameter in the CONFIG section of "formation_control.launch.py".


Inside each folder will be created a '_csv_file' folder containing the data (x, y) of the last simulation. 
In order to see the plots run the "plot_csv.py" by properly setting the PLOT_TASK option in the CONFIG section.

# How to run a simulation

Go inside the folder relative to the task you want to execute
ex cd/task_2.2/
Open the terminal here:
> source /opt/ros/foxy/setup.bash 
> colcon build --symlink-install
> . install/setup.bash
> ros2 launch formation_control formation_control.launch.py