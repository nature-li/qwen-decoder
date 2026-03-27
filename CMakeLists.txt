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

macro(add_cuda_target target)
  add_executable(${target} ${ARGN})
  target_link_libraries(${target} PRIVATE CUDA::cudart cublas)
  target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR})
  set_target_properties(${target} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endmacro()

add_executable(cpu_decoder cpu_main.cpp gguf_loader.cpp common.cpp decoder.cpp cpu_decoder.cpp)

add_cuda_target(gpu_decoder_v1
    gpu_v1/gpu_decoder.cu decoder.cpp common.cpp gguf_loader.cpp)
add_cuda_target(gpu_decoder_v2
    gpu_v2/gpu_decoder.cu decoder.cpp common.cpp gguf_loader.cpp)
add_cuda_target(gpu_decoder_v3
    gpu_v3/gpu_decoder.cu decoder.cpp common.cpp gguf_loader.cpp)