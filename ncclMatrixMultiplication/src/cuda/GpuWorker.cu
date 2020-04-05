#include "GpuWorker.cuh"

template <class Toperation>
GpuWorker<Toperation>::GpuWorker(int gpuRankWorld,int gpuRankSystem)
{
    this->gpuRankWorld=gpuRankWorld;
    this->gpuRankSystem=gpuRankSystem;
}

template <class Toperation>
int GpuWorker<Toperation>::getGpuRankWorld()
{
    return gpuRankWorld;
}
template <class Toperation>
int GpuWorker<Toperation>::getGpuRankSystem()
{
    return gpuRankSystem;
}

template class GpuWorker<double>;