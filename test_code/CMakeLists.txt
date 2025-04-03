# Modified from https://github.com/leimao/CUDA-GEMM-Optimization/blob/main/CMakeLists.txt

cmake_minimum_required (VERSION 3.18)

project(Test_code LANGUAGES CXX CUDA)

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

include_directories("./")

file(GLOB src_file
    "${CMAKE_SOURCE_DIR}/*.cpp"    
    "${CMAKE_SOURCE_DIR}/*.cu"
)

message(STATUS "src_file = ${src_file}")

add_executable(test_main main_shfl_down_sync.cu)
target_link_libraries(test_main CUDA::cublas)

set_target_properties(test_main PROPERTIES 
    CXX_STANDARD 17
    CUDA_STANDARD 17
    CUDA_ARCHITECTURES 89
)
