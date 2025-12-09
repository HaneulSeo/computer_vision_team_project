cmake_minimum_required(VERSION 3.10)
project(project4)

find_package(OpenCV REQUIRED)

add_executable(result project4.cpp)
target_link_libraries(result ${OpenCV_LIBS})
