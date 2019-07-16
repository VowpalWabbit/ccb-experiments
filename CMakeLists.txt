cmake_minimum_required(VERSION 3.5)

project(ccb_experiments C CXX)
set(CMAKE_CXX_STANDARD 11)

set(VW_INSTALL OFF CACHE BOOL "")
set(BUILD_TESTS OFF CACHE BOOL "")
set(BUILD_JAVA OFF CACHE BOOL "")
set(BUILD_PYTHON OFF CACHE BOOL "")
set(BUILD_DOCS OFF CACHE BOOL "")
set(WARNINGS OFF CACHE BOOL "")
add_subdirectory(vowpal_wabbit EXCLUDE_FROM_ALL)

add_executable(ccb_experiments main.cpp)
target_link_libraries(ccb_experiments PRIVATE vw)
