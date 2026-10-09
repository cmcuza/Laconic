cmake_minimum_required(VERSION 3.15)
project(batch_pyserf CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)      # -O2; a debug build would measure the wrong thing
endif()

# The Serf checkout this links against. It is NOT vendored here: only the small
# wrapper below is ours, and pinning a whole third-party tree into this repo to
# build one optional measurement module is not worth it.
set(SERF_ROOT "" CACHE PATH "Path to a built Serf checkout (must contain build/src/libserf.a)")
if(NOT EXISTS "${SERF_ROOT}/build/src/libserf.a")
  message(FATAL_ERROR "SERF_ROOT=${SERF_ROOT} has no build/src/libserf.a - build Serf first")
endif()

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(batch_pyserf wrapper.cpp)
target_include_directories(batch_pyserf PRIVATE "${SERF_ROOT}/src")
target_link_libraries(batch_pyserf PRIVATE "${SERF_ROOT}/build/src/libserf.a")
