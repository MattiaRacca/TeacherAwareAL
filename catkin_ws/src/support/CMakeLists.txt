cmake_minimum_required(VERSION 2.8.3)
project(support)

find_package(catkin REQUIRED COMPONENTS
  message_generation
)

add_service_files(
  FILES
  speech.srv
  led.srv
)

add_message_files(
  FILES
  ExperimentTrigger.msg
)

generate_messages()

catkin_package(CATKIN_DEPENDS
  message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

