## FOLDER organization

The folder contains the following subfolders:
- /task_2.2/: formation control with collision avoidance 
                in order to change the desidered configuration modify the "formation_control.launch.py"

- /task_2.3/: formation control with moving leaders
                in order to change the motion of the leader select the desired dynamics in "the_agent.py" file
                (linear trajectory, circular trajectory o motion to target position)
                to change the input or the target position change relative section in "formation_control.launch.py"

- /task_2.4/: formation control with moving leader and obstacle avoidance
                in order to change the obstacle configuration modify the "formation_control.launch.py" 

Inside each folder there is a /_csv_file/ folder that contains the data of the last simulation. 
In order to see the plots run the "plot_csv.py" by properly setting the Task you want to show.


# RUN THE SIMULATIONS

Go inside the folder relative to the task you want to execute
ex cd/task_2.2/
Open the terminal here:
> source /opt/ros/foxy/setup.bash 
> colcon build --symlink-install
> . install/setup.bash
> ros2 launch formation_control formation_control.launch.py