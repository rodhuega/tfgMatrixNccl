cmake_minimum_required(VERSION 3.8)

# set the project name and version
project(ncclComunicators LANGUAGES CXX CUDA VERSION 1.0)

# add the executable
add_executable(ncclComunicadores ncclComunicadores.cu)

# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to 
# particles will also build with -std=c++11
target_compile_features(ncclComunicadores PUBLIC cxx_std_11)

target_link_libraries(ncclComunicadores PRIVATE cublas nccl blas)

