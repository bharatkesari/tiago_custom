# tiago_custom
Custom ROS package to run planning/learning experiments with Tiago Gallium robot in simulation. Built for ROS Noetic.

## Tiago Gallium Tutorials
Follow these tutorial to install the packages to run the Gazebo simulation of the Tiago Gallium robot: http://wiki.ros.org/Robots/TIAGo/Tutorials

## Running Simulation with Custom .world Files
Copy the .world file to the .../tiago_public_ws/src/pal_gazebo_worlds/worlds/ directory. Then you can simulate the robot in the world by passing the name to the "world" argument when launching. For example, to simulate the robot in "prenovelty_domain.world", run:

```
roslaunch tiago_gazebo tiago_gazebo.launch public_sim:=true end_effector:=pal-gripper world:=prenovelty_domain
``` 

## Teleoping the Robot
After launching a simulation, run

```
rosrun key_teleop key_teleop.py
```

## Mapping, Navigation, and Localization Stack
Follow the following tutorial: http://wiki.ros.org/Robots/TIAGo/Tutorials/Navigation/Mapping. Include the world argument to the launch file if you wish to change the world. For example: 

```
roslaunch tiago_2dnav_gazebo tiago_mapping.launch public_sim:=true world:=prenovelty_domain
```

Save the map using 

```
rosservice call /pal_map_manager/save_map "directory: ''"
```

This will create a directory containing the map files to the ~/.pal/tiago_maps/configurations directory. The map directory will be names as a timestamp. Change the name of the map directory to the name of the .world file that was mapped.

## Changing the Gripper
Pass a gripper name to the end_effector argument to the launch file. Gripper names are located here: http://wiki.ros.org/Robots/TIAGo/Tutorials/Installation/Testing_simulation.

