cmake_minimum_required(VERSION 3.10.0)

# if (NOT CMAKE_CUDA_COMPILER)
#     set(CMAKE_CUDA_COMPILER "/usr/local/cuda-10.2/bin/nvcc")
# endif()

project(ncclMatrixMultiplication LANGUAGES C CUDA CXX VERSION 0.1.0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# enable_language(CUDA)
find_package(CUDA)


file(GLOB cpp_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/*.cpp" )
file(GLOB c_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/c/*.c" )
# add_executable(ncclMatrixMultiplication ${cpp_SRC})




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

cuda_add_executable(ncclMatrixMultiplication ${c_SRC} ${cpp_SRC} ${CU_O})
target_compile_features(ncclMatrixMultiplication PUBLIC cxx_std_11)
# target_link_libraries(ncclMatrixMultiplication PRIVATE cublas nccl)
target_link_libraries(ncclMatrixMultiplication cublas curand nccl blas)
set_target_properties(ncclMatrixMultiplication PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)