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

## Mapping
Follow the tutorial: http://wiki.ros.org/Robots/TIAGo/Tutorials/Navigation/Mapping. Include the world argument to the launch file if you wish to change the world. For example: 

```
roslaunch tiago_2dnav_gazebo tiago_mapping.launch public_sim:=true world:=prenovelty_domain
```

Save the map using 

```
rosservice call /pal_map_manager/save_map "directory: ''"
```

This will create a directory containing the map files to the ~/.pal/tiago_maps/configurations directory. The map directory will be names as a timestamp. Change the name of the map directory to the name of the .world file that was mapped.

## Navigation
Follow the tutorial: http://wiki.ros.org/Robots/TIAGo/Tutorials/Navigation/Localization. An example launch command would be: 

```
roslaunch tiago_2dnav_gazebo tiago_navigation.launch public_sim:=true world:=prenovelty_domain
```

This will only work if there exists a directory with all required map files in ~/.pal/tiago_maps/configurations. Here is an example of launching with a map directory in a different directory

```
roslaunch tiago_2dnav_gazebo tiago_navigation.launch public_sim:=true lost:=true map:=$HOME/tiago_public_ws/src/tiago_custom/maps/prenovelty_domain
```

## Changing the Gripper
Pass a gripper name to the end_effector argument to the launch file. Gripper names are located here: http://wiki.ros.org/Robots/TIAGo/Tutorials/Installation/Testing_simulation.

## Configurations
### objects.yaml
Contains a dictionary called 'objects'. Within the dictionary, Gazebo object names are keys, and values are sub-dictionaries. Each sub-dictionary holds the keys 'offset' and 'direction'. These parameters define a point relative to the Gazebo object used as the goal point when navigating to the object. The 'offset' and 'direction' parameters are defined in the 'map' frame.

### waypoints.yaml
Contains a dictionary called 'waypoints'. The keys of the dictionary are strings naming each waypoint. The values of the dictionary are arrays defining the location and orientation of each waypoint (xyzrpy). The waypoints are relative to the 'map' frame.

### boundaries.yaml
Contains two dictionaries used to maintain the 'at' and 'facing' predicate of the robot. The dictionary 'at_boundaries' has string keys naming each boundary, and the values are arrays containing a set of (x, y) coordinates defining a polygon for the boundary. The dictionary 'facing_boundaries' contains a sub-dictionary for each object, or abstract thing the robot can face in the world. Each sub-dictionary defines an radius and an angle for the robot to be facing that thing. If the thing is abstract, a central location must be provided in the sub-dictionary. 

### transforms.yaml
Contains a dictionary called 'transforms'. The keys of the dictionary are strings representing the names of reference frames. The values of the dictionary are arrays defining transformation between the new frame and the 'map' frame.

## Custom Service Types
### ObjPose.srv
The request is comprised of a string, and the response returns a geometry_msgs/Point and a Boolean.

### StringBool.srv
The request is comprised of a string, and the response returns a Boolean.

### TwistBool.srv
The request is comprised of a geometry_msgs/Twist, and returns a Boolean.

## Custom Services
### get_object_pose (ObjPose)
Returns the location of an object in Gazebo. The name of the object is sent as the request. If the object is found, the 'success' field of the response is set to true. If the object is not found, the 'success' field of the response is set to false.

### obj_nav (StringBool)
Navigates the robot to an object in Gazebo. The object must have an entry in objects.yaml. Returns true if navigation is successful and false otherwise.

### waypoints_nav (StringBool)
Navigates the robot to a waypoint defined in waypoints.yaml. Returns true if navigation is successful and false otherwise.

### pose_nav (TwistBool)
Navigates the robot to a point in the ‘map’ frame.  Returns true if navigation is successful and false otherwise.

### pick (Empty)
TBD

### place (Empty)
TBD