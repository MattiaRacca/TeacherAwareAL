cmake_minimum_required(VERSION 2.8.3)
project(control_gui)

find_package(catkin REQUIRED COMPONENTS)

catkin_python_setup()
catkin_package()

install(FILES plugin.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
