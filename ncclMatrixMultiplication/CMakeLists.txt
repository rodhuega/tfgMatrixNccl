cmake_minimum_required(VERSION 3.10.0)

# if (NOT CMAKE_CUDA_COMPILER)
#     set(CMAKE_CUDA_COMPILER "/usr/local/cuda-10.2/bin/nvcc")
# endif()

project(ncclMatrixMultiplication LANGUAGES CUDA CXX VERSION 0.1.0)

# enable_language(CUDA)

file(GLOB cpp_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/*.cpp" )
add_executable(ncclMatrixMultiplication ${cpp_SRC})



target_compile_features(ncclMatrixMultiplication PUBLIC cxx_std_11)
target_link_libraries(ncclMatrixMultiplication PRIVATE cublas nccl blas)
set_target_properties(ncclMatrixMultiplication PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# COMPILE CU FILES
file(GLOB cuda_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.cpp")
# list( APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_30,code=sm_30; -std=c++11")
# CUDA_COMPILE(CU_O ${CUDA_FILES})

# SETUP FOR CPP FILES
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/cpp)


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)