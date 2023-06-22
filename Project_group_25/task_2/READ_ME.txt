## FOLDER organization

The folder contains the following subfolders:
- /task_2.2/: formation control with collision avoidance 
                in order to change the desidered configuration modify the CONFIG section in "formation_control.launch.py"
                e.g. change the number of agents (N) or get a 3d formation (_3d_formation)

- /task_2.3/: formation control with moving leaders
                to change the desired leaders' motion modify the TYPE_MOTION parameter in the CONFIG section of "formation_control.launch.py".

- /task_2.4/: formation control with obstacle avoidance
                to change the number of obstacles modify the N_OBSTACLES parameter in the CONFIG section of "formation_control.launch.py".


Inside each folder will be created a '_csv_file' folder containing the data (x, y) of the last simulation.  
In order to see the plots run the "plot_csv.py" by properly setting the PLOT_TASK option in the CONFIG section, please pay attention to insert the right path to "_csv_file" folder.


# How to run a simulation

Go inside the folder relative to the task you want to execute
ex cd/task_2.2/
Open the terminal here:
> source /opt/ros/foxy/setup.bash 
> colcon build --symlink-install
> . install/setup.bash
> ros2 launch formation_control formation_control.launch.py
