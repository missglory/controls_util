cmake_minimum_required(VERSION 2.8)

project(controls_util)

find_package( OpenCV REQUIRED )
add_subdirectory("dlib" dlib_build)

add_executable(${PROJECT_NAME} "main.cpp")

target_link_libraries(${PROJECT_NAME} dlib::dlib ${OpenCV_LIBS})
