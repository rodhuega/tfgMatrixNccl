#include "MatrixMain.cuh"

template <class Toperation>
MatrixMain<Toperation>::MatrixMain(NcclMultiplicationEnvironment<Toperation>* ncclMultEnv,std::string id,int rows,int columns)
{
    this->ncclMultEnv=ncclMultEnv;
    this->id=id;
    this->rowsReal=rows;
    this->columnsReal=columns;
    this->isMatrixHostHere=false;
    this->isDistributed=false;
    this->deleteMatrixHostAtDestroyment=false;
    this->deleteObjectAtDestroyment=true;
    this->hostMatrix=nullptr;
    if(id=="")
    {
        this->id=this->ncclMultEnv->generateRandomId();
    }
}

template <class Toperation>
MatrixMain<Toperation>::MatrixMain(NcclMultiplicationEnvironment<Toperation>* ncclMultEnv,std::string id,int rows,int columns,Toperation* matrix):MatrixMain(ncclMultEnv,id,rows,columns)
{
    this->ncclMultEnv=ncclMultEnv;
    this->hostMatrix=matrix;
    this->isMatrixHostHere=true;
}

template <class Toperation>
MatrixMain<Toperation>::~MatrixMain()
{
    if(deleteObjectAtDestroyment)
    {
        if(isMatrixHostHere && deleteMatrixHostAtDestroyment)
        {
            MatrixUtilities<Toperation>::matrixFree(hostMatrix);
        }
        deleteGpuWorkers();
    }
}

template <class Toperation>
std::string  MatrixMain<Toperation>::getId()
{
    return id;
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
int MatrixMain<Toperation>::getMeshColumnSize()
{
    return meshColumnSize;
}

template <class Toperation>
int MatrixMain<Toperation>::getMeshRowSize()
{
    return meshRowSize;
}

template <class Toperation>
bool MatrixMain<Toperation>::getIsMatrixHostHere()
{
    return isMatrixHostHere;
}

template <class Toperation>
bool MatrixMain<Toperation>::getDeleteMatrixHostAtDestroyment()
{
    return deleteMatrixHostAtDestroyment;
}

template <class Toperation>
Toperation *MatrixMain<Toperation>::getHostMatrix()
{
    if(!isMatrixHostHere)
    {
        recoverMatrixToHost();
    }
    return hostMatrix;
}
template <class Toperation>
std::vector<GpuWorker<Toperation>*> MatrixMain<Toperation>::getGpuWorkers()
{
    return gpuWorkers;
}

template <class Toperation>
void MatrixMain<Toperation>::setId(std::string id)
{
    this->id=id;
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
void MatrixMain<Toperation>::setDeleteMatrixHostAtDestroyment(bool deleteMatrixHostAtDestroyment)
{
    this->deleteMatrixHostAtDestroyment=deleteMatrixHostAtDestroyment;
}


template <class Toperation>
void MatrixMain<Toperation>::setMatrixOperationProperties(int meshRowSize, int meshColumnSize, int blockRowSize, int blockColumnSize)
{
    this->isDistributed=false;
    gpuWorkers.clear();
    this->meshRowSize=meshRowSize;
    this->meshColumnSize=meshColumnSize;
    this->blockRowSize=blockRowSize;
    this->blockColumnSize=blockColumnSize;
    this->numberOfRowBlocks = ceil(this->rowsUsed / this->blockRowSize);
    this->numberOfColumnBlocks = ceil(this->columnsUsed / this->blockColumnSize);
    this->numberOfTotalBlocks = this->numberOfRowBlocks * this->numberOfColumnBlocks;
    this->blockSize = this->blockRowSize * this->blockColumnSize;
    blocksInitialPosition.resize(numberOfTotalBlocks);
    int i, posColumnBelong, posRowBelong,indexBlock,gpuRealId;
    for (i = 0,indexBlock=0; i < numberOfTotalBlocks; i++)
    {
        posColumnBelong = (i % numberOfColumnBlocks) * rowsReal * blockColumnSize;
        posRowBelong = (i / numberOfColumnBlocks) * blockRowSize;
        blocksInitialPosition[i]=(posColumnBelong + posRowBelong);
        //Debido a ColumnMajorOrder corrijo al indice del bloque que pertenece para una correcta formación de la malla.
        indexBlock=(indexBlock+numberOfColumnBlocks);
        if(indexBlock>=numberOfTotalBlocks){
            indexBlock%=(numberOfTotalBlocks-1);
        }
        //Creacion de los gpuWorkers y su primer bloque
        if(i<ncclMultEnv->getGpuSizeOperationWorld())
        {
            gpuRealId=MatrixUtilitiesCuda<Toperation>::getRealGpuId(i,ncclMultEnv->getGpuSizeSystem());
            CUDACHECK(cudaSetDevice(gpuRealId));
            GpuWorker<Toperation> *gpuW= new GpuWorker<Toperation>(i,gpuRealId,this);
            gpuWorkers.push_back(gpuW);
            cudaStream_t *newStream = new cudaStream_t;
            CUDACHECK(cudaStreamCreate(newStream));
            gpuWorkers[i]->addStream(newStream);
            Toperation *newMatrix=MatrixUtilitiesCuda<Toperation>::cudaMatrixMemoryAllocation(blockRowSize,blockColumnSize,newStream);
            gpuWorkers[i]->addMatrixLocal(newMatrix);
        }
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
    if(!isDistributed)
    {
        int i,j,k,blockColumnSizeCopy,blockRowSizeCopy;
        for(i=0;i<ncclMultEnv->getGpuSizeOperationWorld()&&i<numberOfTotalBlocks;i++)
        {
            CUDACHECK(cudaSetDevice(gpuWorkers[i]->getGpuRankSystem()));
            for(j=i;j<numberOfTotalBlocks;j+=ncclMultEnv->getGpuSizeOperationWorld())
            {
                Toperation *newMatrix;
                cudaStream_t *newStream;
                if(j!=i)//El primer bloque ya estaba creado de la llamada a setMatrixOperationProperties
                {
                    newStream = new cudaStream_t;
                    CUDACHECK(cudaStreamCreate(newStream));
                    gpuWorkers[i]->addStream(newStream);
                    newMatrix=MatrixUtilitiesCuda<Toperation>::cudaMatrixMemoryAllocation(blockRowSize,blockColumnSize,newStream);
                    gpuWorkers[i]->addMatrixLocal(newMatrix);
                }else 
                {
                    newStream=gpuWorkers[i]->getStream(0);
                    newMatrix=gpuWorkers[i]->getMatrixLocal(0);
                }
                blockColumnSizeCopy = calculateBlockDimensionToCopy(calculateColumnColor(i), numberOfColumnBlocks, blockColumnSize, columnsUsed, columnsReal);
                blockRowSizeCopy = calculateBlockDimensionToCopy(calculateRowColor(i), numberOfRowBlocks, blockRowSize, rowsUsed, rowsReal);
                for(k=0;k<blockColumnSizeCopy;k++)
                {
                    CUDACHECK(cudaMemcpyAsync(&newMatrix[k*blockRowSize],&hostMatrix[blocksInitialPosition[j]+k*rowsReal],blockRowSizeCopy*sizeof(Toperation),cudaMemcpyHostToDevice,*newStream));
                }
            }
    }
    setIsDistributed(true);
    }
}

template <class Toperation>
void MatrixMain<Toperation>::recoverMatrixToHost()
{
    if(!isMatrixHostHere)
    {
        int i,j,k,blockColumnSizeCopy,blockRowSizeCopy,matrixLocalIndex;
        hostMatrix=MatrixUtilities<Toperation>::matrixMemoryAllocation(rowsReal,columnsReal);
        for(i=0;i<ncclMultEnv->getGpuSizeOperationWorld()&&i<numberOfTotalBlocks;i++)
        {
            CUDACHECK(cudaSetDevice(gpuWorkers[i]->getGpuRankSystem()));
            for(j=i,matrixLocalIndex=0;j<numberOfTotalBlocks;j+=ncclMultEnv->getGpuSizeOperationWorld(),matrixLocalIndex++)
            {
                Toperation *newMatrix;
                cudaStream_t *newStream;
                newStream=gpuWorkers[i]->getStream(matrixLocalIndex);
                newMatrix=gpuWorkers[i]->getMatrixLocal(matrixLocalIndex);
                blockColumnSizeCopy = calculateBlockDimensionToCopy(calculateColumnColor(i), numberOfColumnBlocks, blockColumnSize, columnsUsed, columnsReal);
                blockRowSizeCopy = calculateBlockDimensionToCopy(calculateRowColor(i), numberOfRowBlocks, blockRowSize, rowsUsed, rowsReal);
                for(k=0;k<blockColumnSizeCopy;k++)
                {
                    CUDACHECK(cudaMemcpyAsync(&hostMatrix[blocksInitialPosition[j]+k*rowsReal],&newMatrix[k*blockRowSize],blockRowSizeCopy*sizeof(Toperation),cudaMemcpyDeviceToHost,*newStream));
                }
            }
        }
        waitAllStreamsOfAllWorkers();
        setIsMatrixHostHere(true);
    }
}

template <class Toperation>
void MatrixMain<Toperation>::deleteGpuWorkers()
{
    int i;
    for(i=0;i<this->gpuWorkers.size();i++)
    {
        delete this->gpuWorkers[i];
    }
    this->gpuWorkers.clear();
}

template <class Toperation>
void MatrixMain<Toperation>::assignationToActualObject(const MatrixMain<Toperation>& B)
{
    deleteGpuWorkers();   
    this->id=B.id;//Creo que no.
    this->ncclMultEnv=B.ncclMultEnv;
    this->hostMatrix=B.hostMatrix;
    this->gpuWorkers=B.gpuWorkers;
    this->deleteMatrixHostAtDestroyment=B.deleteMatrixHostAtDestroyment;
    this->blocksInitialPosition=B.blocksInitialPosition;
    this->rowsReal=B.rowsReal;
    this->rowsUsed=B.rowsUsed;
    this->columnsReal=B.columnsReal;
    this->columnsUsed=B.columnsUsed;
    this->isDistributed=B.isDistributed;
    this->isMatrixHostHere=B.isMatrixHostHere;
    this->blockRowSize=B.blockRowSize;
    this->blockColumnSize=B.blockColumnSize;
    this->blockSize=B.blockSize;    
    this->meshRowSize=B.meshRowSize;
    this->meshColumnSize=B.meshColumnSize;
    this->numberOfRowBlocks=B.numberOfRowBlocks;
    this->numberOfColumnBlocks=B.numberOfColumnBlocks;
    this->numberOfTotalBlocks=B.numberOfTotalBlocks;
}

template <class Toperation>
MatrixMain<Toperation>& MatrixMain<Toperation>::operator*=(MatrixMain<Toperation>& B )
{
    if(this->id==B.id)
    {
        B.deleteObjectAtDestroyment=false;
        MatrixMain<Toperation>* res;
        {
            MatrixMain<Toperation> aux =B;
            res=&ncclMultEnv->performCalculations(*this,aux,"");
        }
        B.deleteObjectAtDestroyment=true;
        assignationToActualObject(*res);
    }else
    {
        MatrixMain<Toperation>& res=ncclMultEnv->performCalculations(*this,B,id);
        assignationToActualObject(res);
    }
    return *this;
}

template <class Toperation>
MatrixMain<Toperation>& MatrixMain<Toperation>::operator*(MatrixMain<Toperation>& B)
{
    if(B.id==this->id)
    {
        B.deleteObjectAtDestroyment=false;
        MatrixMain<Toperation>* res;
        {
            MatrixMain<Toperation> aux =B;
            res=&ncclMultEnv->performCalculations(*this,aux,"");
        }
        B.deleteObjectAtDestroyment=true;
        return *res;
    }
    return ncclMultEnv->performCalculations(*this,B,"");
}

template <class Toperation>
MatrixMain<Toperation>& MatrixMain<Toperation>::operator=(const MatrixMain<Toperation>& B)
{
    assignationToActualObject(B);
    return *this;
}



template class MatrixMain<double>;
template class MatrixMain<float>;