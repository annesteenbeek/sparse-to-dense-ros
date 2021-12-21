cmake_minimum_required(VERSION 2.8.3)
project(sparse_to_dense)

find_package(catkin REQUIRED COMPONENTS
rospy
std_msgs
sensor_msgs
message_generation
dynamic_reconfigure
)

add_message_files(
    FILES
    Result.msg
    SampleMetrics.msg
)

catkin_python_setup()

generate_messages(
    DEPENDENCIES
    std_msgs
)

generate_dynamic_reconfigure_options(
  cfg/SparseToDense.cfg
)

# add_dependencies(sparse_to_dense ${PROJECT_NAME}_gencfg)