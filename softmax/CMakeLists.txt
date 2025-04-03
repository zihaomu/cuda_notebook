# Modified from https://github.com/leimao/CUDA-GEMM-Optimization/blob/main/CMakeLists.txt

cmake_minimum_required (VERSION 3.18)

project(CUDA_SOFT_MAX LANGUAGES CXX CUDA)

#=======================================================================

find_package(CUDAToolkit REQUIRED)


# TODO add option to enable/disable cublas

set(CUTLASS_DIR "E:\\my_project\\cutlass-3.8.0")
include_directories(${CUTLASS_DIR}/include)
# find_path(CUDA_GEMM_INCLUDE_DIRS cuda_gemm.hpp HINTS ${CMAKE_SOURCE_DIR}/include)

set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CUTLASS_NVCC_ARCHS 89)

include_directories("${CMAKE_SOURCE_DIR}/src")
include_directories("${CMAKE_SOURCE_DIR}/../utils")

file(GLOB src_file
    "${CMAKE_SOURCE_DIR}/src/*.cpp"  
    "${CMAKE_SOURCE_DIR}/src/*.cu"
    "${CMAKE_SOURCE_DIR}/../utils/*.cu"
    "${CMAKE_SOURCE_DIR}/../utils/*.cpp"
)

message(STATUS "src_file = ${src_file}")

add_library(cuda_softmax
    SHARED
    ${src_file}
)

target_link_libraries(cuda_softmax CUDA::cublas CUDA::cudart)

add_executable(softmax_main_fp32 softmax_main.cpp)
target_link_libraries(softmax_main_fp32 cuda_softmax CUDA::cublas)

# add_executable(gemm_main_fp16 main_fp16.cpp)
# target_link_libraries(gemm_main_fp16 cuda_gemm CUDA::cublas)

