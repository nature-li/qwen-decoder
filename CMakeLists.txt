cmake_minimum_required(VERSION 3.18)
project(qwen-decoder CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

add_executable(gguf_info main.cpp gguf_loader.cpp common.cpp cpu_decoder.cpp)