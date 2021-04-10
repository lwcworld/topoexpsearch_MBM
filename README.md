
# About
- provides service node which return exploratory-search path 

# tested environtment
- ubuntu 18.04
- python 2.7
- ROS melodic
- gazebo 9
- keras 2.4.3
- tensorflow 2.3.0

# Node
## Service (EST_server)
### input arguments
- map (nav_msgs/OccupancyGrid)
- node_values (sensor_msgs/PointCloud)
- start (geometry_msgs/Point32)

### return
- path (nav_msgs/Path)

# How to run
- EST server \
``` rosrun topoexpsearch_planner srv_EST.py ```

- server call example \
``` call_EST.py ```
