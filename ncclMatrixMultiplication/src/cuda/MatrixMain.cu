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
}

template <class Toperation>
void MatrixMain<Toperation>::distributeMatrixIntoGpus()
{
    int i;
    for(i=0;i<ncclMultEnv->getGpuSizeOperationWorld();i++)
    {
        int gpuRealId=MatrixUtilitiesCuda<Toperation>::getRealGpuId(i,ncclMultEnv->getGpuSizeSystem());
        GpuWorker<Toperation> *gpuW= new GpuWorker<Toperation>(i,gpuRealId);
        gpuWorkers.push_back(gpuW);
    }
    for(i=0;i<ncclMultEnv->getGpuSizeOperationWorld();i++)
    {
        
    }
}


template class MatrixMain<double>;