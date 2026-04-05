cmake_minimum_required(VERSION 3.18)
project(qwen-decoder CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
# set(CMAKE_CUDA_ARCHITECTURES 89)
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

# gRPC + Protobuf
find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)
message(STATUS "gRPC: ${gRPC_VERSION}")
message(STATUS "Protobuf: ${Protobuf_VERSION}")

# 生成 proto
set(PROTO_FILE ${CMAKE_SOURCE_DIR}/proto/pd_service.proto)
set(PROTO_DIR ${CMAKE_SOURCE_DIR}/proto)
set(PROTO_SRCS ${PROTO_DIR}/pd_service.pb.cc)
set(PROTO_HDRS ${PROTO_DIR}/pd_service.pb.h)
set(GRPC_SRCS ${PROTO_DIR}/pd_service.grpc.pb.cc)
set(GRPC_HDRS ${PROTO_DIR}/pd_service.grpc.pb.h)

add_custom_command(
    OUTPUT ${PROTO_SRCS} ${PROTO_HDRS} ${GRPC_SRCS} ${GRPC_HDRS}
    COMMAND protoc
    ARGS --grpc_out=${PROTO_DIR}
         --cpp_out=${PROTO_DIR}
         --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
         -I${PROTO_DIR}
         ${PROTO_FILE}
    DEPENDS ${PROTO_FILE}
    COMMENT "Generating gRPC sources from ${PROTO_FILE}"
)

add_library(pd_proto STATIC ${PROTO_SRCS} ${GRPC_SRCS})
target_link_libraries(pd_proto PUBLIC gRPC::grpc++ protobuf::libprotobuf)
target_include_directories(pd_proto PUBLIC ${PROTO_DIR})

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
      CUDA::cudart cublas ${NCCL_LIB}
      pd_proto gRPC::grpc++ protobuf::libprotobuf)
  target_include_directories(${target} PRIVATE
      ${CMAKE_SOURCE_DIR} ${NCCL_INCLUDE} ${PROTO_DIR})
  set_target_properties(${target} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endmacro()

add_cuda_target(gpu_decoder_v14
    gpu_v14/gpu_decoder.cu
    gpu_v14/gpu_main.cu
    decoder.cpp
    common.cpp
    gguf_loader.cpp
    scheduler.cpp
    kv_cache.cpp
    prefix_cache.cpp)

add_pd_target(prefill_node
    gpu_v14/prefill_main.cu
    gpu_v14/gpu_decoder.cu
    gpu_v14/pd_comm.cpp
    decoder.cpp
    common.cpp
    gguf_loader.cpp
    scheduler.cpp
    kv_cache.cpp
    prefix_cache.cpp)

add_pd_target(decode_node
    gpu_v14/decode_main.cu
    gpu_v14/gpu_decoder.cu
    gpu_v14/pd_comm.cpp
    decoder.cpp
    common.cpp
    gguf_loader.cpp
    scheduler.cpp
    kv_cache.cpp
    prefix_cache.cpp)