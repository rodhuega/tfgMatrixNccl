#include "GpuWorker.cuh"

template <class Toperation>
GpuWorker<Toperation>::GpuWorker(int gpuRankWorld,int gpuRankSystem,MatrixMain<Toperation>* matrixMainGlobal)
{
    this->gpuRankWorld=gpuRankWorld;
    this->gpuRankSystem=gpuRankSystem;
    this->matrixMainGlobal=matrixMainGlobal;
}

template <class Toperation>
GpuWorker<Toperation>::GpuWorker(const GpuWorker<Toperation> &gpuW)
{
    this->gpuRankWorld=gpuW.gpuRankWorld;
    this->gpuRankSystem=gpuW.gpuRankSystem;
    this->matrixMainGlobal=gpuW.matrixMainGlobal;
    int i;
    CUDACHECK(cudaSetDevice(this->gpuRankSystem));
    cudaStream_t *newStream;
    Toperation *auxMatrix;
    for(i=0;i<gpuW.gpuMatricesLocal.size();i++)
    {
        newStream = new cudaStream_t;
        CUDACHECK(cudaStreamCreate(newStream));
        auxMatrix=MatrixUtilitiesCuda<Toperation>::cudaMatrixMemoryAllocationGPU(this->matrixMainGlobal->getBlockRowSize(),this->matrixMainGlobal->getBlockColumnSize(),newStream);
        CUDACHECK(cudaMemcpyAsync(auxMatrix,gpuW.gpuMatricesLocal[i],this->matrixMainGlobal->getBlockRowSize()*this->matrixMainGlobal->getBlockColumnSize()*sizeof(Toperation),cudaMemcpyDeviceToDevice,*newStream));
        streams.push_back(newStream);
        this->gpuMatricesLocal.push_back(auxMatrix);
    }
}

template <class Toperation>
GpuWorker<Toperation>::~GpuWorker()
{
    int i;
    CUDACHECK(cudaSetDevice(gpuRankSystem));
    for(i=0;i<gpuMatricesLocal.size();i++)
    {
        MatrixUtilitiesCuda<Toperation>::matrixFreeGPU(gpuMatricesLocal[i]);
        CUDACHECK(cudaStreamDestroy(*streams[i]));
    }
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
std::vector<Toperation *> GpuWorker<Toperation>::getMatricesLocal()
{
    return gpuMatricesLocal;
}

template <class Toperation>
cudaStream_t *GpuWorker<Toperation>::getStream(int pos)
{
    return streams[pos];
}

template <class Toperation>
std::vector<cudaStream_t*> GpuWorker<Toperation>::getStreams()
{
    return streams;
}

template <class Toperation>
void GpuWorker<Toperation>::addMatrixLocal(Toperation *gpumatrixLocal)
{
    gpuMatricesLocal.push_back(gpumatrixLocal);
}

template <class Toperation>
void GpuWorker<Toperation>::addStream(cudaStream_t* stream)
{
    streams.push_back(stream);
}

template <class Toperation>
void GpuWorker<Toperation>::waitAllStreams()
{
    int i;
    for(i=0;i<streams.size();i++)
    {
        CUDACHECK(cudaStreamSynchronize(*streams[i]));
    }
}

template class GpuWorker<double>;
template class GpuWorker<float>;