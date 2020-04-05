#include "GpuWorker.cuh"

template <class Toperation>
GpuWorker<Toperation>::GpuWorker(int gpuRankWorld,int gpuRankSystem,MatrixMain<Toperation>* matrixMainGlobal)
{
    this->gpuRankWorld=gpuRankWorld;
    this->gpuRankSystem=gpuRankSystem;
    this->matrixMainGlobal=matrixMainGlobal;
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

template <class Toperation>
Toperation *GpuWorker<Toperation>::getMatrixLocal(int pos)
{
    return gpuMatricesLocal[pos];
}

template <class Toperation>
void GpuWorker<Toperation>::setMatrixLocal(Toperation *gpumatrixLocal)
{
    gpuMatricesLocal.push_back(gpumatrixLocal);
}

template class GpuWorker<double>;