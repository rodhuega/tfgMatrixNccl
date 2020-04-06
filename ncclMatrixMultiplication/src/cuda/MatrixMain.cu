#include "MatrixMain.cuh"

template <class Toperation>
MatrixMain<Toperation>::MatrixMain(NcclMultiplicationEnvironment<Toperation>* ncclMultEnv,std::string id,int rows,int columns)
{
    this->ncclMultEnv=ncclMultEnv;
    this->id=id;
    this->rowsReal=rows;
    this->columnsReal=columns;
    this->isDistributed=false;
    this->ncclMultEnv->setOrAddMatrixMain(id,this);
}

template <class Toperation>
MatrixMain<Toperation>::MatrixMain(NcclMultiplicationEnvironment<Toperation>* ncclMultEnv,std::string id,int rows,int columns,Toperation* matrix):MatrixMain(ncclMultEnv,id,rows,columns)
{
    this->ncclMultEnv=ncclMultEnv;
    this->hostMatrix=matrix;
    this->isMatrixHostHere=true;
}

template <class Toperation>
int MatrixMain<Toperation>::getRowsReal()
{
    return rowsReal;
}

template <class Toperation>
int MatrixMain<Toperation>::getColumnsReal()
{
    return columnsReal;
}

template <class Toperation>
int MatrixMain<Toperation>::getRowsUsed()
{
    return rowsUsed;
}

template <class Toperation>
int MatrixMain<Toperation>::getColumnsUsed()
{
    return columnsUsed;
}

template <class Toperation>
bool MatrixMain<Toperation>::getIsDistributed()
{
    return isDistributed;
}

template <class Toperation>
int MatrixMain<Toperation>::getBlockSize()
{
    return blockSize;
}

template <class Toperation>
int MatrixMain<Toperation>::getBlockRowSize()
{
    return blockRowSize;
}

template <class Toperation>
int MatrixMain<Toperation>::getBlockColumnSize()
{
    return blockColumnSize;
}

template <class Toperation>
bool MatrixMain<Toperation>::getIsMatrixHostHere()
{
    return isMatrixHostHere;
}

template <class Toperation>
Toperation *MatrixMain<Toperation>::getHostMatrix()
{
    return hostMatrix;
}
template <class Toperation>
std::vector<GpuWorker<Toperation>*> MatrixMain<Toperation>::getGpuWorkers()
{
    return gpuWorkers;
}

template <class Toperation>
void MatrixMain<Toperation>::setRowsUsed(int rowsUsed)
{
    this->rowsUsed = rowsUsed;
}

template <class Toperation>
void MatrixMain<Toperation>::setColumnsUsed(int columnsUsed)
{
    this->columnsUsed = columnsUsed;
}

template <class Toperation>
void MatrixMain<Toperation>::setIsDistributed(bool isDistributed)
{
    this->isDistributed = isDistributed;
}

template <class Toperation>
void MatrixMain<Toperation>::setIsMatrixHostHere(bool isMatrixHostHere)
{
    this->isMatrixHostHere = isMatrixHostHere;
}

template <class Toperation>
void MatrixMain<Toperation>::setBlockAndMeshSize(int meshRowSize, int meshColumnSize, int blockRowSize, int blockColumnSize)
{
    this->meshRowSize=meshRowSize;
    this->meshColumnSize=meshColumnSize;
    this->blockRowSize=blockRowSize;
    this->blockColumnSize=blockColumnSize;
    this->numberOfRowBlocks = ceil(this->rowsUsed / this->blockRowSize);
    this->numberOfColumnBlocks = ceil(this->columnsUsed / this->blockColumnSize);
    this->numberOfTotalBlocks = this->numberOfRowBlocks * this->numberOfColumnBlocks;
    this->blockSize = this->blockRowSize * this->blockColumnSize;

    int i, posColumnBelong, posRowBelong;
    for (i = 0; i < numberOfTotalBlocks; i++)
    {
        posRowBelong = (i / meshColumnSize) * columnsReal * blockRowSize;
        posColumnBelong = (i % meshColumnSize) * blockColumnSize;
        blocksInitialPosition.push_back(posColumnBelong + posRowBelong);
    }
}

template <class Toperation>
int MatrixMain<Toperation>::calculateRowColor(int gpuRank)
{
    return gpuRank / numberOfColumnBlocks;
}

template <class Toperation>
int MatrixMain<Toperation>::calculateColumnColor(int gpuRank)
{
    return gpuRank % numberOfColumnBlocks;
}

template <class Toperation>
int MatrixMain<Toperation>::calculateBlockDimensionToCopy(int color, int meshDimensionSize, int blockDimenensionSize, int dimensionUsed, int dimensionReal)
{
    return (color != (meshDimensionSize - 1)) ? blockDimenensionSize : (blockDimenensionSize - (dimensionUsed - dimensionReal));
}

template <class Toperation>
void MatrixMain<Toperation>::waitAllStreamsOfAllWorkers()
{
    int i;
    for(i=0;i<gpuWorkers.size();i++)
    {
        CUDACHECK(cudaSetDevice(gpuWorkers[i]->getGpuRankSystem()));
        gpuWorkers[i]->waitAllStreams();
    }
}

template <class Toperation>
void MatrixMain<Toperation>::distributeMatrixIntoGpus()
{
    int i,j,k,blockColumnSizeCopy,blockRowSizeCopy;
    for(i=0;i<ncclMultEnv->getGpuSizeOperationWorld();i++)
    {
        int gpuRealId=MatrixUtilitiesCuda<Toperation>::getRealGpuId(i,ncclMultEnv->getGpuSizeSystem());
        GpuWorker<Toperation> *gpuW= new GpuWorker<Toperation>(i,gpuRealId,this);
        gpuWorkers.push_back(gpuW);
        CUDACHECK(cudaSetDevice(gpuWorkers[i]->getGpuRankSystem()));
        for(j=0;j<numberOfTotalBlocks;j+=ncclMultEnv->getGpuSizeOperationWorld())
        {
            cudaStream_t *newStream = new cudaStream_t;
            CUDACHECK(cudaStreamCreate(newStream));
            gpuWorkers[i]->addStream(newStream);
            Toperation *newMatrix=MatrixUtilitiesCuda<Toperation>::cudaMatrixMemoryAllocation(blockRowSize,blockColumnSize,newStream);
            
            blockColumnSizeCopy = calculateBlockDimensionToCopy(calculateColumnColor(i), numberOfColumnBlocks, blockColumnSize, columnsUsed, columnsReal);
            blockRowSizeCopy = calculateBlockDimensionToCopy(calculateRowColor(i), numberOfRowBlocks, blockRowSize, rowsUsed, rowsReal);
            for(k=0;k<blockColumnSizeCopy;k++)
            {
                //W.I.P Indice blocksInitialPosition puede que este mal
                CUDACHECK(cudaMemcpyAsync(&newMatrix[k*blockRowSize],&hostMatrix[blocksInitialPosition[i]+k*columnsReal],blockRowSizeCopy*sizeof(Toperation),cudaMemcpyHostToDevice,*newStream));
            }
            gpuWorkers[i]->setMatrixLocal(newMatrix);
        }
    }
}


template class MatrixMain<double>;