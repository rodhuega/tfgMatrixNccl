cmake_minimum_required(VERSION 3.8)

# set the project name and version
project(test LANGUAGES CXX CUDA VERSION 1.0)

find_package(CUDA)
find_package(Matlab)
CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/Matrix.h)


matlab_add_mex(NAME probandoMex SRC call_gpu.cu)


# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to 
# particles will also build with -std=c++11
target_compile_features(probandoMex PUBLIC cxx_std_11)

# target_link_libraries(probandoMex PRIVATE cublas)


##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## Ready for compiler gcc 4.9                            %
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#CUDA_LIB_PATH=/usr/local/cuda/lib64
#CUDA_LIB=cuda
#CUDA_RT=cudart
#CUBLAS=cublas
#MATLAB_INCLUDE_PATH=/usr/local/MATLAB/R2015b/extern/include
#
#call_gpu.mexa64: call_gpu.o
#	mex call_gpu.o -cxx -L$(CUDA_LIB_PATH) -l$(CUDA_LIB) -l$(CUDA_RT) -l$(CUBLAS)
#
#call_gpu.o: call_gpu.cu Matrix.h
#	nvcc --std=c++11 -c call_gpu.cu -I$(MATLAB_INCLUDE_PATH) -arch=compute_50 --compiler-options -fPIC
#
#clean:
#	rm -f call_gpu.mexa64 call_gpu.o
