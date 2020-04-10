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
    this->hostMatrix=nullptr;
    if(id=="")
    {
        this->id=this->ncclMultEnv->generateRandomId();
    }
    this->ncclMultEnv->setOrAddMatrixMain(this->id,this);
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
    if(isMatrixHostHere && deleteMatrixHostAtDestroyment)
    {
        MatrixUtilities<Toperation>::matrixFree(hostMatrix);
    }
    int i;
    for(i=0;i<gpuWorkers.size();i++)
    {
        delete gpuWorkers[i];
    }
    gpuWorkers.clear();
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
    //Quitar la anterior id del entorno
    this->ncclMultEnv->removeMatrixMain(this->id,false);
    this->id=id;
    //Agregar la nueva id al entorno
    this->ncclMultEnv->setOrAddMatrixMain(id,this);
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
            GpuWorker<Toperation> *gpuW= new GpuWorker<Toperation>(i,gpuRealId,this);
            gpuWorkers.push_back(gpuW);
            cudaStream_t *newStream = new cudaStream_t;
            CUDACHECK(cudaStreamCreate(newStream));
            gpuWorkers[i]->addStream(newStream);
            Toperation *newMatrix=MatrixUtilitiesCuda<Toperation>::cudaMatrixMemoryAllocation(blockRowSize,blockColumnSize,newStream);
            gpuWorkers[i]->setMatrixLocal(newMatrix);
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
                gpuWorkers[i]->setMatrixLocal(newMatrix);
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

template <class Toperation>
void MatrixMain<Toperation>::recoverMatrixToHost()
{
    //OJO QUE SI LA MTRIZ TIENE EN UN WORKER VARIAS MATRICESLOCALES NO VA BIEN. O ESO CREO. TAMPOCO SE SI SE PUEDE DAR EL CASO
    int i,j,k,blockColumnSizeCopy,blockRowSizeCopy;
    hostMatrix=MatrixUtilities<Toperation>::matrixMemoryAllocation(rowsReal,columnsReal);
    for(i=0;i<ncclMultEnv->getGpuSizeOperationWorld()&&i<numberOfTotalBlocks;i++)
    {
        CUDACHECK(cudaSetDevice(gpuWorkers[i]->getGpuRankSystem()));
        for(j=0;j<numberOfTotalBlocks;j+=ncclMultEnv->getGpuSizeOperationWorld())
        {
            Toperation *newMatrix;
            cudaStream_t *newStream;
            newStream=gpuWorkers[i]->getStream(0);
            newMatrix=gpuWorkers[i]->getMatrixLocal(0);
            blockColumnSizeCopy = calculateBlockDimensionToCopy(calculateColumnColor(i), numberOfColumnBlocks, blockColumnSize, columnsUsed, columnsReal);
            blockRowSizeCopy = calculateBlockDimensionToCopy(calculateRowColor(i), numberOfRowBlocks, blockRowSize, rowsUsed, rowsReal);
            for(k=0;k<blockColumnSizeCopy;k++)
            {
                CUDACHECK(cudaMemcpyAsync(&hostMatrix[blocksInitialPosition[i]+k*rowsReal],&newMatrix[k*blockRowSize],blockRowSizeCopy*sizeof(Toperation),cudaMemcpyDeviceToHost,*newStream));
            }
        }
    }
    waitAllStreamsOfAllWorkers();
    setIsMatrixHostHere(true);
}

template <class Toperation>
MatrixMain<Toperation> MatrixMain<Toperation>::operator*=(MatrixMain<Toperation> B )
{
    /////////////////NO FUNCIONA////////////////////////
    MatrixMain<Toperation> aux=(*this)*B;
    return aux;
}

template <class Toperation>
MatrixMain<Toperation> MatrixMain<Toperation>::operator*(MatrixMain<Toperation> B)
{
    return *(ncclMultEnv->performCalculations(id,B.getId(),""));
}



template class MatrixMain<double>;
template class MatrixMain<float>;