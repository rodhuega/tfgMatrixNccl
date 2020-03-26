set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER "")
set(CMAKE_CUDA_HOST_LINK_LAUNCHER "/usr/local/bin/g++")
set(CMAKE_CUDA_COMPILER_ID "NVIDIA")
set(CMAKE_CUDA_COMPILER_VERSION "10.0.130")
set(CMAKE_CUDA_STANDARD_COMPUTED_DEFAULT "03")
set(CMAKE_CUDA_COMPILE_FEATURES "cuda_std_03;cuda_std_11;cuda_std_14")
set(CMAKE_CUDA03_COMPILE_FEATURES "cuda_std_03")
set(CMAKE_CUDA11_COMPILE_FEATURES "cuda_std_11")
set(CMAKE_CUDA14_COMPILE_FEATURES "cuda_std_14")
set(CMAKE_CUDA17_COMPILE_FEATURES "")
set(CMAKE_CUDA20_COMPILE_FEATURES "")

set(CMAKE_CUDA_PLATFORM_ID "Linux")
set(CMAKE_CUDA_SIMULATE_ID "")
set(CMAKE_CUDA_COMPILER_FRONTEND_VARIANT "")
set(CMAKE_CUDA_SIMULATE_VERSION "")


set(CMAKE_CUDA_COMPILER_ENV_VAR "CUDACXX")
set(CMAKE_CUDA_HOST_COMPILER_ENV_VAR "CUDAHOSTCXX")

set(CMAKE_CUDA_COMPILER_LOADED 1)
set(CMAKE_CUDA_COMPILER_ID_RUN 1)
set(CMAKE_CUDA_SOURCE_FILE_EXTENSIONS cu)
set(CMAKE_CUDA_LINKER_PREFERENCE 15)
set(CMAKE_CUDA_LINKER_PREFERENCE_PROPAGATES 1)

set(CMAKE_CUDA_SIZEOF_DATA_PTR "8")
set(CMAKE_CUDA_COMPILER_ABI "ELF")
set(CMAKE_CUDA_LIBRARY_ARCHITECTURE "x86_64-linux-gnu")

if(CMAKE_CUDA_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_CUDA_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_CUDA_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_CUDA_COMPILER_ABI}")
endif()

if(CMAKE_CUDA_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "x86_64-linux-gnu")
endif()

set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "/usr/local/cuda/targets/x86_64-linux/include")

set(CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES "rt;pthread;dl")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES "/usr/local/cuda/targets/x86_64-linux/lib/stubs;/usr/local/cuda/targets/x86_64-linux/lib")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES "/opt/intel/compilers_and_libraries_2019.4.243/linux/ipp/include;/opt/intel/compilers_and_libraries_2019.4.243/linux/mkl/include;/opt/intel/compilers_and_libraries_2019.4.243/linux/pstl/include;/opt/intel/compilers_and_libraries_2019.4.243/linux/tbb/include;/opt/intel/compilers_and_libraries_2019.4.243/linux/daal/include;/usr/local/include/c++/5.3.0;/usr/local/include/c++/5.3.0/x86_64-unknown-linux-gnu;/usr/local/include/c++/5.3.0/backward;/usr/local/lib/gcc/x86_64-unknown-linux-gnu/5.3.0/include;/usr/local/include;/usr/local/lib/gcc/x86_64-unknown-linux-gnu/5.3.0/include-fixed;/usr/include/x86_64-linux-gnu;/usr/include")
set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES "rt;pthread;dl;stdc++;m;gcc_s;gcc;c;gcc_s;gcc")
set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "/usr/local/cuda/targets/x86_64-linux/lib/stubs;/usr/local/cuda/targets/x86_64-linux/lib;/usr/local/lib/gcc/x86_64-unknown-linux-gnu/5.3.0;/usr/local/lib64;/lib/x86_64-linux-gnu;/lib64;/usr/lib/x86_64-linux-gnu;/opt/intel/compilers_and_libraries_2019.4.243/linux/mpi/intel64/libfabric/lib;/opt/intel/compilers_and_libraries_2019.4.243/linux/ipp/lib/intel64;/opt/intel/compilers_and_libraries_2019.4.243/linux/compiler/lib/intel64_lin;/opt/intel/compilers_and_libraries_2019.4.243/linux/mkl/lib/intel64_lin;/opt/intel/compilers_and_libraries_2019.4.243/linux/tbb/lib/intel64/gcc4.7;/opt/intel/compilers_and_libraries_2019.4.243/linux/daal/lib/intel64_lin;/opt/intel/compilers_and_libraries_2019.4.243/linux/tbb/lib/intel64_lin/gcc4.4;/usr/local/lib")
set(CMAKE_CUDA_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_LINKER "/usr/bin/ld")
set(CMAKE_MT "")
