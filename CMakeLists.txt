cmake_minimum_required(VERSION 3.18)
project(qwen-decoder CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_ARCHITECTURES 100)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

find_package(CUDAToolkit REQUIRED)

# NCCL
find_library(NCCL_LIB nccl HINTS /usr/lib/x86_64-linux-gnu /usr/local/lib)
find_path(NCCL_INCLUDE nccl.h HINTS /usr/include /usr/local/include)
if(NOT NCCL_LIB OR NOT NCCL_INCLUDE)
  message(FATAL_ERROR "NCCL not found")
endif()

# ZMQ
find_library(ZMQ_LIB zmq HINTS /usr/local/lib /usr/lib)
find_path(ZMQ_INCLUDE zmq.hpp HINTS /usr/local/include /usr/include)
if(NOT ZMQ_LIB OR NOT ZMQ_INCLUDE)
  message(FATAL_ERROR "ZMQ not found")
endif()
message(STATUS "ZMQ: ${ZMQ_LIB}")

macro(add_cuda_target target)
  add_executable(${target} ${ARGN})
  target_link_libraries(${target} PRIVATE CUDA::cudart cublas)
  target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR})
  set_target_properties(${target} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endmacro()

macro(add_pd_target target)
  add_executable(${target} ${ARGN})
  target_link_libraries(${target} PRIVATE
      CUDA::cudart cublas ${NCCL_LIB} ${ZMQ_LIB})
  target_include_directories(${target} PRIVATE
      ${CMAKE_SOURCE_DIR} ${NCCL_INCLUDE} ${ZMQ_INCLUDE})
  set_target_properties(${target} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endmacro()

add_cuda_target(gpu_decoder_v12
    gpu_v12/gpu_decoder.cu
    gpu_v12/gpu_main.cu
    decoder.cpp 
    common.cpp 
    gguf_loader.cpp 
    scheduler.cpp 
    kv_cache.cpp
    prefix_cache.cpp)

add_pd_target(prefill_node
    gpu_v12/prefill_main.cu
    gpu_v12/gpu_decoder.cu
    gpu_v12/pd_comm.cpp
    decoder.cpp 
    common.cpp 
    gguf_loader.cpp 
    scheduler.cpp 
    kv_cache.cpp
    prefix_cache.cpp)

add_pd_target(decode_node
    gpu_v12/decode_main.cu
    gpu_v12/gpu_decoder.cu
    gpu_v12/pd_comm.cpp
    decoder.cpp 
    common.cpp 
    gguf_loader.cpp 
    scheduler.cpp 
    kv_cache.cpp
    prefix_cache.cpp)