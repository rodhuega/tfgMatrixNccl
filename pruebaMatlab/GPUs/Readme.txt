make -f Makefile_mex
mkdir build && cd build && cmake .. -DBUILD_FOR_MATLAB=ON -DMatlab_ROOT_DIR="/usr/local/MATLAB/R2018a" && make call_gpu
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2 /usr/local/MATLAB/R2018a/bin/matlab -nodesktop
LD_PRELOAD="/usr/local/lib64/libstdc++.so.6" matlab
addpath(genpath('/home/rodhuega/directorioTrabajo/tfgMatrixNccl/pruebaMatlab'));testexp(7000,1000,7000);
addpath(genpath('/home/rodhuega/directorioTrabajo/tfgMatrixNccl/pruebaMatlab'));testexp(1000,1000,30000);
