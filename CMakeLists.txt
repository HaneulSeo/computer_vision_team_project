cmake_minimum_required(VERSION 3.10)
project(project4)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(result project4.cpp)
target_link_libraries(result ${OpenCV_LIBS})
