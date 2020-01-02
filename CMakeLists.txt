SET(execName LANE)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(${execName})

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 OLD)
	cmake_policy(SET CMP0015 OLD)
endif(COMMAND cmake_policy)

SET(EXECUTABLE_OUTPUT_PATH ".")

find_package(OpenCV REQUIRED)


list(APPEND OpenCV_LINKED_COMPONENTS ${OpenCV_LIBS})

include_directories(${OPENCV_INCLUDE_DIRS})

link_directories(${OpenCV_LIBRARY_DIRS})

ADD_EXECUTABLE(${execName} "main.cpp")

add_definitions(-std=c++11 -g -O3)

TARGET_LINK_LIBRARIES(${execName} ${OpenCV_LIBRARIES})
