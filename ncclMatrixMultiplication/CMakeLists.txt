cmake_minimum_required(VERSION 3.10.0)

# if (NOT CMAKE_CUDA_COMPILER)
#     set(CMAKE_CUDA_COMPILER "/usr/local/cuda-10.2/bin/nvcc")
# endif()

project(ncclMatrixMultiplication LANGUAGES C CUDA CXX VERSION 0.1.0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fpic")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
# enable_language(CUDA)
find_package(CUDA)
find_package(MKL REQUIRED)



file(GLOB versus1Gpu_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/versus1Gpu/*.cpp" )
file(GLOB versus1GpuAndXt_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/versus1GpuAndXt/*.cu" )
file(GLOB c_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/c/*.c" )
# add_executable(ncclMatrixMultiplication ${cpp_SRC})
option(BUILD_FOR_MATLAB "Build the matlab library mex" OFF)



# COMPILE CU FILES
file(GLOB cuda_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.cu")
# list( APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_30,code=sm_30; -std=c++11")
CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include/cuda)
CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include/cpp)
cuda_compile(CU_O ${cuda_SRC})

# SETUP FOR CPP FILES
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/c)
include_directories("${CUDA_INCLUDE_DIRS}")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/cuda)

cuda_add_library(ncclMatrixMultiplication ${CU_O})
target_compile_features(ncclMatrixMultiplication PUBLIC cxx_std_11)
# target_link_libraries(ncclMatrixMultiplication PRIVATE cublas nccl)
target_link_libraries(ncclMatrixMultiplication cublas curand nccl mkl_intel_lp64 mkl_sequential mkl_core)
set_target_properties(ncclMatrixMultiplication PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

cuda_add_executable(versus1Gpu ${versus1Gpu_SRC} ${c_SRC})
target_link_libraries(versus1Gpu ncclMatrixMultiplication)
target_compile_features(versus1Gpu PUBLIC cxx_std_11)

cuda_add_executable(versus1GpuAndXt ${versus1GpuAndXt_SRC} ${c_SRC})
target_link_libraries(versus1GpuAndXt ncclMatrixMultiplication)
target_compile_features(versus1GpuAndXt PUBLIC cxx_std_11)

if(BUILD_FOR_MATLAB)
    find_package(Matlab)
    file(GLOB matlab_SRC "${CMAKE_CURRENT_SOURCE_DIR}/matlab/src/*.cu" )
    matlab_add_mex(NAME call_gpu SRC ${matlab_SRC} LINK_TO ncclMatrixMultiplication)
endif(BUILD_FOR_MATLAB)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)