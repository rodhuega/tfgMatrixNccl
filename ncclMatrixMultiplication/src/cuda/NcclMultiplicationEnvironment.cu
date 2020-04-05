#include "NcclMultiplicationEnvironment.cuh"


template <class Toperation>
NcclMultiplicationEnvironment<Toperation>::NcclMultiplicationEnvironment(int gpuSizeWorld,int gpuRoot,OperationType opType)
{
    this->gpuRoot=gpuRoot;

    CUDACHECK(cudaGetDeviceCount(&gpuSizeSystem));

    if(gpuSizeWorld!=-1)
    {
        this->gpuSizeWorld=gpuSizeWorld;
    }else
    {
        this->gpuSizeWorld=gpuSizeSystem;
    }
    this->opType=opType;
    if(opType==MultDouble)
    {
        basicOperationType=ncclDouble;
    }else
    {
        basicOperationType=ncclFloat;
    }
    std::cout<<"System gpus: "<<this->gpuSizeSystem<<". World gpus: "<<this->gpuSizeWorld<<std::endl;
}


template class NcclMultiplicationEnvironment<double>;
