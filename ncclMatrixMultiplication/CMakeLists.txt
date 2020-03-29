cmake_minimum_required(VERSION 3.10.0)

# if (NOT CMAKE_CUDA_COMPILER)
#     set(CMAKE_CUDA_COMPILER "/usr/local/cuda-10.2/bin/nvcc")
# endif()

project(ncclMatrixMultiplication VERSION 0.1.0)

enable_language(CUDA)


add_executable(ncclMatrixMultiplication ./src/cpp/main.cpp)

# target_compile_features(ncclMatrixMultiplication PUBLIC cxx_std_11)
target_link_libraries(ncclMatrixMultiplication PRIVATE cublas nccl blas)
set_target_properties(ncclMatrixMultiplication PROPERTIES CUDA_SEPARABLE_COMPILATION ON)



set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)