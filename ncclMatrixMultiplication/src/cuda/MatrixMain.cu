#include "MatrixMain.cuh"

template <class Toperation>
MatrixMain<Toperation>::MatrixMain(NcclMultiplicationEnvironment<Toperation>* ncclMultEnv,int rows,int columns)
{
    this->ncclMultEnv=ncclMultEnv;
    this->rowsReal=rows;
    this->columnsReal=columns;
    this->isMatrixHostHere=false;
    this->isDistributed=false;
    this->deleteMatrixHostAtDestroyment=false;
    this->deleteObjectAtDestroyment=true;
    this->hostMatrix=nullptr;
    this->alphaGemm=1;
}

template <class Toperation>
MatrixMain<Toperation>::MatrixMain(NcclMultiplicationEnvironment<Toperation>* ncclMultEnv,int rows,int columns,Toperation* matrix):MatrixMain(ncclMultEnv,rows,columns)
{
    this->ncclMultEnv=ncclMultEnv;
    this->hostMatrix=matrix;
    this->isMatrixHostHere=true;
}

template <class Toperation>
MatrixMain<Toperation>::MatrixMain(const MatrixMain<Toperation> &maMain)
{
    assignationToActualObject(maMain,true);
}

template <class Toperation>
MatrixMain<Toperation>::MatrixMain(MatrixMain<Toperation> &&B)
{
    this->ncclMultEnv=std::move(B.ncclMultEnv);
    this->alphaGemm=std::move(B.alphaGemm);
    this->deleteObjectAtDestroyment=std::move(B.deleteObjectAtDestroyment);
    this->deleteMatrixHostAtDestroyment=std::move(B.deleteMatrixHostAtDestroyment);
    this->blocksInitialPosition=std::move(B.blocksInitialPosition);
    this->blocksInitialPositionDiagonal=std::move(blocksInitialPositionDiagonal);
    this->rowsReal=std::move(B.rowsReal);
    this->rowsUsed=std::move(B.rowsUsed);
    this->columnsReal=std::move(B.columnsReal);
    this->columnsUsed=std::move(B.columnsUsed);
    this->isDistributed=std::move(B.isDistributed);
    this->isMatrixHostHere=std::move(B.isMatrixHostHere);
    this->blockRowSize=std::move(B.blockRowSize);
    this->blockColumnSize=std::move(B.blockColumnSize);
    this->blockSize=std::move(B.blockSize);    
    this->meshRowSize=std::move(B.meshRowSize);
    this->meshColumnSize=std::move(B.meshColumnSize);
    this->numberOfRowBlocks=std::move(B.numberOfRowBlocks);
    this->numberOfColumnBlocks=std::move(B.numberOfColumnBlocks);
    this->numberOfTotalBlocks=std::move(B.numberOfTotalBlocks);  
    this->hostMatrix=std::move(B.hostMatrix);
    this->gpuWorkers=std::move(B.gpuWorkers);
}


template <class Toperation>
MatrixMain<Toperation>::~MatrixMain()
{
    if(deleteObjectAtDestroyment)
    {
        if(isMatrixHostHere && deleteMatrixHostAtDestroyment)
        {
            MatrixUtilitiesCuda<Toperation>::matrixFreeCPU(hostMatrix);
        }
        deleteGpuWorkers();
        blocksInitialPosition.clear();
        blocksInitialPositionDiagonal.clear();
    }
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
Toperation MatrixMain<Toperation>::getAlphaGemm()
{
    return alphaGemm;
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
void MatrixMain<Toperation>::getHostMatrixInThisPointer(Toperation* pointerMatrix)
{
    getHostMatrix();
    memcpy(pointerMatrix,hostMatrix,sizeof(Toperation)*rowsReal*columnsReal);
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
void MatrixMain<Toperation>::setAlphaGemm(Toperation alphaGemm)
{
    this->alphaGemm = alphaGemm;
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
    if(!isMatrixHostHere && hostMatrix!=nullptr)
    {
        MatrixUtilitiesCuda<Toperation>::matrixFreeCPU(hostMatrix);
        hostMatrix=nullptr;
    }
}

template <class Toperation>
void MatrixMain<Toperation>::setMatrixHost(Toperation* newMatrixHost)
{
    if(isMatrixHostHere)
    {
        MatrixUtilitiesCuda<Toperation>::matrixFreeCPU(hostMatrix);
        hostMatrix=nullptr;
    }
    if(isDistributed)
    {
        deleteGpuWorkers();
    }
    this->hostMatrix=newMatrixHost;
    this->isMatrixHostHere=true;
    this->isDistributed=false;
}

template <class Toperation>
void MatrixMain<Toperation>::setMatrixHostToFullValue(Toperation valueForHost)
{
    if(isMatrixHostHere)
    {
        MatrixUtilitiesCuda<Toperation>::matrixFreeCPU(hostMatrix);
        hostMatrix=nullptr;
    }
    isMatrixHostHere=true;
    hostMatrix=MatrixUtilitiesCuda<Toperation>::matrixMemoryAllocationCPU(rowsReal,columnsReal);
    int i;
    for(i=0;i<rowsReal*columnsReal;i++)
    {
        hostMatrix[i]=valueForHost;
    }
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
    blocksInitialPositionDiagonal.resize(numberOfTotalBlocks);
    int i, posColumnBelongColumnMajorOrder, posRowBelongColumnMajorOrder,gpuRealId,posColumnBelongRowMajorOrder, posRowBelongRowMajorOrder,rowMajorOrderIndex,actualIndexDiagonal=0;
    int minimumDimensionSize=min(blockRowSize,blockColumnSize);
    int diagonalBlockSize=0,diagonalBlockFirtsPosition;
    for (i = 0;i < numberOfTotalBlocks; i++)
    {
        //Cálculo del índice global de la posición inicial de cada bloque
        posColumnBelongColumnMajorOrder = (i % numberOfColumnBlocks) * rowsReal * blockColumnSize;
        posRowBelongColumnMajorOrder = (i / numberOfColumnBlocks) * blockRowSize;
        blocksInitialPosition[i]=(posColumnBelongColumnMajorOrder + posRowBelongColumnMajorOrder);
        //Cálculo de los elementos para la diagonal
        posRowBelongRowMajorOrder = (i / meshColumnSize) * columnsReal * blockRowSize;
        posColumnBelongRowMajorOrder = (i % meshColumnSize) * blockColumnSize;
        rowMajorOrderIndex=+posRowBelongRowMajorOrder+posColumnBelongRowMajorOrder;
        if(actualIndexDiagonal>=blocksInitialPosition[i] && actualIndexDiagonal<(blocksInitialPosition[i]+blockRowSize))
        {//Caso de que este en la primera columna 
            diagonalBlockFirtsPosition=actualIndexDiagonal-blocksInitialPosition[i];
            if(i==0)
            {
                diagonalBlockSize=minimumDimensionSize;
                blocksInitialPositionDiagonal[i]=std::make_tuple(actualIndexDiagonal,0,diagonalBlockSize);

            }else{
                diagonalBlockSize=blockRowSize-diagonalBlockFirtsPosition;
                blocksInitialPositionDiagonal[i]=std::make_tuple(actualIndexDiagonal,blockRowSize-diagonalBlockSize,min(diagonalBlockSize,minimumDimensionSize));
            }
            actualIndexDiagonal+=(columnsReal+1)*min(diagonalBlockSize,minimumDimensionSize);
        }else if(actualIndexDiagonal>=rowMajorOrderIndex && actualIndexDiagonal<(rowMajorOrderIndex+blockColumnSize))
        {//Caso de que no este en la primera columna pero este en la primera fila
            diagonalBlockFirtsPosition=actualIndexDiagonal-rowMajorOrderIndex;
            diagonalBlockSize=blockColumnSize-diagonalBlockFirtsPosition;
            blocksInitialPositionDiagonal[i]=std::make_tuple(actualIndexDiagonal,(blockColumnSize-diagonalBlockSize)*blockRowSize,min(diagonalBlockSize,minimumDimensionSize));
            actualIndexDiagonal+=(columnsReal+1)*min(diagonalBlockSize,minimumDimensionSize);
        }else
        {
            blocksInitialPositionDiagonal[i]=std::make_tuple(-1,-1,-1);
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
            Toperation *newMatrix=MatrixUtilitiesCuda<Toperation>::cudaMatrixMemoryAllocationGPU(blockRowSize,blockColumnSize,newStream);
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
        if(!isMatrixHostHere)
        {
            throw std::invalid_argument("No existe matriz en el host, asi que no se puede distribuir");
        }
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
                    newMatrix=MatrixUtilitiesCuda<Toperation>::cudaMatrixMemoryAllocationGPU(blockRowSize,blockColumnSize,newStream);
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
void MatrixMain<Toperation>::distributeMatrixMySelfIntoGpus()
{
    if(!isMatrixHostHere)
    {
        throw std::invalid_argument("No existe matriz en el host, asi que no se puede distribuir");
    }
    OperationProperties op = MatrixUtilitiesCuda<Toperation>::getMeshAndMatrixSize(getRowsReal(), getColumnsReal(), getRowsReal(), getColumnsReal(), this->ncclMultEnv->getGpuSizeWorld());
    setRowsUsed(op.rowsA);
    setColumnsUsed(op.columnsAorRowsB);
    this->ncclMultEnv->setGpuSizeOperationWorld(op.gpuSize);
    this->ncclMultEnv->setGpuSizeOperationSystem(min(this->ncclMultEnv->getGpuSizeSystem(),op.gpuSize));
    setMatrixOperationProperties(op.meshRowSize,op.meshColumnSize,op.blockRowSizeA,op.blockColumnSizeA);
    distributeMatrixIntoGpus();
    waitAllStreamsOfAllWorkers();
    setIsDistributed(true);
}

template <class Toperation>
void MatrixMain<Toperation>::recoverMatrixToHost()
{
    if(!isMatrixHostHere)
    {
        if(gpuWorkers.size()==0)
        {
            throw std::invalid_argument("La matriz no se encuentra distribuida, asi que no se puede recuperar.");
        }
        int i,j,k,blockColumnSizeCopy,blockRowSizeCopy,matrixLocalIndex;
        hostMatrix=MatrixUtilitiesCuda<Toperation>::matrixMemoryAllocationCPU(rowsReal,columnsReal);
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
void MatrixMain<Toperation>::axpy(const Toperation& alpha,MatrixMain<Toperation>& X,MatrixMain<Toperation>& Y)
{
    if(Y.rowsReal!=X.rowsReal || Y.columnsReal!=X.columnsReal)
    {
        throw std::invalid_argument("Las matrices no son del mismo tamaño y no se puede realizar la operación");
    }
    if(!Y.isDistributed || !X.isDistributed)
    {
        double square= sqrt(Y.ncclMultEnv->getGpuSizeOperationWorld());
        bool isPerfectSquare=(square-floor(square))==0;
        if(!Y.isDistributed && isPerfectSquare)
        {
            Y.distributeMatrixMySelfIntoGpus();
        }
        if(!X.isDistributed && isPerfectSquare)
        {
            X.distributeMatrixMySelfIntoGpus();
        }
    }
    if(Y.isDistributed && X.isDistributed && Y.blockColumnSize==X.blockColumnSize && Y.blockRowSize==X.blockRowSize)
    {
        OperationType opType= Y.ncclMultEnv->getOperationType();
        int i,j,idPhysicGpu,matrixLocalIndex;
        for(i=0;i<Y.ncclMultEnv->getGpuSizeOperationWorld()&&i<Y.numberOfTotalBlocks;i++)
        {
            idPhysicGpu=Y.gpuWorkers[i]->getGpuRankSystem();
            CUDACHECK(cudaSetDevice(idPhysicGpu));
            for(j=i,matrixLocalIndex=0;j<Y.numberOfTotalBlocks;j+=Y.ncclMultEnv->getGpuSizeOperationWorld(),matrixLocalIndex++)
            {
                MatrixUtilitiesCuda<Toperation>::axpyCublas(Y.ncclMultEnv->getCublasHandlers()[idPhysicGpu],opType,Y.blockRowSize*Y.blockColumnSize ,X.gpuWorkers[i]->getMatrixLocal(matrixLocalIndex),Y.gpuWorkers[i]->getMatrixLocal(matrixLocalIndex),alpha,1,1);
            }
        }
        Y.ncclMultEnv->waitAllCublasStreams();
        setIsMatrixHostHere(false);
    }else 
    {
        throw std::invalid_argument("Ambas matrices no están distribuidas o sus bloques no son de las mismas dimensiones");
    }
}

template <class Toperation>
void MatrixMain<Toperation>::axpy(const Toperation& alpha,MatrixMain<Toperation>& X)
{
    axpy(alpha,X,*this);
}

template <class Toperation>
Toperation MatrixMain<Toperation>::norm1()
{
    if(!isDistributed)
    {
        distributeMatrixMySelfIntoGpus();
    }
    OperationType opType= ncclMultEnv->getOperationType();
    int i,j,k,matrixLocalIndex,idPhysicGpu,columnColor;
    Toperation res;
    //vector que almacena la suma de sus columnas
    Toperation** columnBlocks = new Toperation*[numberOfTotalBlocks];
    Toperation* maximumOfEachColumnColor = new Toperation[numberOfColumnBlocks];
    //Cada bloque suma sus columnas
    for(i=0;i<ncclMultEnv->getGpuSizeOperationWorld()&&i<numberOfTotalBlocks;i++)
    {
        idPhysicGpu=gpuWorkers[i]->getGpuRankSystem();
        CUDACHECK(cudaSetDevice(idPhysicGpu));
        for(j=i,matrixLocalIndex=0;j<numberOfTotalBlocks;j+=ncclMultEnv->getGpuSizeOperationWorld(),matrixLocalIndex++)
        {
            columnBlocks[j]=(Toperation*)malloc(sizeof(Toperation)*this->blockColumnSize);
            for(k=0;k<this->blockColumnSize;k++)
            {
                MatrixUtilitiesCuda<Toperation>::sumCublas(ncclMultEnv->getCublasHandlers()[idPhysicGpu],opType,this->blockRowSize,1,&gpuWorkers[i]->getMatrixLocal(matrixLocalIndex)[k*this->blockRowSize],&columnBlocks[j][k]);
            }
        }

    }
    ncclMultEnv->waitAllCublasStreams();
    //Se calcula el maximo de cada color de columna
    for(i=0;i<numberOfTotalBlocks;i++)
    {
        columnColor=this->calculateColumnColor(i);
        if(i!=columnColor)
        {
            MatrixUtilitiesCuda<Toperation>::axpyBlas(opType,this->blockColumnSize,columnBlocks[i],columnBlocks[columnColor],1,1,1);
        }
    }
    //Se busca el maximo en la rowColor0 de todas las columnas ya sumadas.
    for(i=0;i<numberOfColumnBlocks;i++)
    {
        maximumOfEachColumnColor[i]=MatrixUtilitiesCuda<Toperation>::maximumBlas(opType,this->blockColumnSize,columnBlocks[i],1);
    }
    res=MatrixUtilitiesCuda<Toperation>::maximumBlas(opType,numberOfColumnBlocks,maximumOfEachColumnColor,1);
    //Liberar recursos
    for(i=0;i<ncclMultEnv->getGpuSizeOperationWorld()&&i<numberOfTotalBlocks;i++)
    {
        idPhysicGpu=gpuWorkers[i]->getGpuRankSystem();
        CUDACHECK(cudaSetDevice(idPhysicGpu));
        for(j=i,matrixLocalIndex=0;j<numberOfTotalBlocks;j+=ncclMultEnv->getGpuSizeOperationWorld(),matrixLocalIndex++)
        {
            free(columnBlocks[j]);
        }
    }
    delete[] columnBlocks;
    delete[] maximumOfEachColumnColor;
    return res;
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
void MatrixMain<Toperation>::assignationToActualObject(const MatrixMain<Toperation>& B,bool deepCopy)
{
    deleteGpuWorkers();
    this->ncclMultEnv=B.ncclMultEnv;
    this->alphaGemm=B.alphaGemm;
    this->deleteObjectAtDestroyment=B.deleteObjectAtDestroyment;
    this->deleteMatrixHostAtDestroyment=B.deleteMatrixHostAtDestroyment;
    this->blocksInitialPosition=B.blocksInitialPosition;
    this->blocksInitialPositionDiagonal=B.blocksInitialPositionDiagonal;
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
    if(deepCopy)
    {
        if(isMatrixHostHere)
        {
            this->hostMatrix=MatrixUtilitiesCuda<Toperation>::matrixMemoryAllocationCPU(this->rowsReal, this->columnsReal);
            memcpy(this->hostMatrix,B.hostMatrix,sizeof(Toperation)*this->rowsReal*this->columnsReal);
        }else
        {
            this->hostMatrix=nullptr;
        }

        if(isDistributed)
        {
            int i;
            GpuWorker<Toperation>* aux;
            for(i=0;i<B.gpuWorkers.size();i++)
            {
                aux= new GpuWorker<Toperation>(*B.gpuWorkers[i]);
                this->gpuWorkers.push_back(aux);
            }
            this->waitAllStreamsOfAllWorkers();
        }

    }else
    {
        this->hostMatrix=B.hostMatrix;
        this->gpuWorkers=B.gpuWorkers;
    }
}

template <class Toperation>
MatrixMain<Toperation>& MatrixMain<Toperation>::operator*=(MatrixMain<Toperation>& B )
{
    if(this==&B)
    {
        MatrixMain<Toperation>* res;
        {
            MatrixMain<Toperation> aux =B;
            aux.setDeleteMatrixHostAtDestroyment(true);
            res=&ncclMultEnv->performCalculations(*this,aux);
        }
        assignationToActualObject(*res,false);
    }else
    {
        MatrixMain<Toperation>& res=ncclMultEnv->performCalculations(*this,B);
        assignationToActualObject(res,false);
    }
    return *this;
}

template <class Toperation>
MatrixMain<Toperation>& MatrixMain<Toperation>::operator*(MatrixMain<Toperation>& B)
{
    if(&B==this)
    {
        MatrixMain<Toperation>* res;
        {
            MatrixMain<Toperation> aux =B;
            aux.setDeleteMatrixHostAtDestroyment(true);
            res=&ncclMultEnv->performCalculations(*this,aux);
        }
        return *res;
    }
    return ncclMultEnv->performCalculations(*this,B);
}

template <class Toperation>
MatrixMain<Toperation>& MatrixMain<Toperation>::operator*=(const Toperation& alpha)
{
    if(!isDistributed)
    {
        distributeMatrixMySelfIntoGpus();
    }
    OperationType opType= ncclMultEnv->getOperationType();
    int i,j,idPhysicGpu;
    for(i=0;i<gpuWorkers.size();i++)
    {
        idPhysicGpu=gpuWorkers[i]->getGpuRankSystem();
        CUDACHECK(cudaSetDevice(idPhysicGpu));
        for(j=0;j<gpuWorkers[i]->getMatricesLocal().size();j++)
        {
            MatrixUtilitiesCuda<Toperation>::scalarCublas(ncclMultEnv->getCublasHandlers()[idPhysicGpu],opType,blockRowSize, blockColumnSize,gpuWorkers[i]->getMatrixLocal(j),alpha,1);
        }
    }
    ncclMultEnv->waitAllCublasStreams();
    setIsMatrixHostHere(false);
    return *this;
}

template <class Toperation>
MatrixMain<Toperation> MatrixMain<Toperation>::operator*(const Toperation& alpha)
{
    MatrixMain<Toperation> aux =*this;
    aux*=alpha;
    return aux;
}
template <class Toperation>
MatrixMain<Toperation>& MatrixMain<Toperation>::operator/=(const Toperation& alpha)
{
    return *this*=(1/alpha);
}

template <class Toperation>
MatrixMain<Toperation> MatrixMain<Toperation>::operator/(const Toperation& alpha)
{
    MatrixMain<Toperation> aux =*this;
    aux/=alpha;
    return aux;
}

template <class Toperation>
MatrixMain<Toperation>& MatrixMain<Toperation>::operator=(const MatrixMain<Toperation>& B)
{
    assignationToActualObject(B,true);
    return *this;
}

template <class Toperation>
MatrixMain<Toperation>& MatrixMain<Toperation>::operator=(MatrixMain<Toperation>&& B)
{
    this->ncclMultEnv=std::move(B.ncclMultEnv);
    this->alphaGemm=std::move(B.alphaGemm);
    this->deleteObjectAtDestroyment=std::move(B.deleteObjectAtDestroyment);
    this->deleteMatrixHostAtDestroyment=std::move(B.deleteMatrixHostAtDestroyment);
    this->blocksInitialPosition=std::move(B.blocksInitialPosition);
    this->blocksInitialPositionDiagonal=std::move(blocksInitialPositionDiagonal);
    this->rowsReal=std::move(B.rowsReal);
    this->rowsUsed=std::move(B.rowsUsed);
    this->columnsReal=std::move(B.columnsReal);
    this->columnsUsed=std::move(B.columnsUsed);
    this->isDistributed=std::move(B.isDistributed);
    this->isMatrixHostHere=std::move(B.isMatrixHostHere);
    this->blockRowSize=std::move(B.blockRowSize);
    this->blockColumnSize=std::move(B.blockColumnSize);
    this->blockSize=std::move(B.blockSize);    
    this->meshRowSize=std::move(B.meshRowSize);
    this->meshColumnSize=std::move(B.meshColumnSize);
    this->numberOfRowBlocks=std::move(B.numberOfRowBlocks);
    this->numberOfColumnBlocks=std::move(B.numberOfColumnBlocks);
    this->numberOfTotalBlocks=std::move(B.numberOfTotalBlocks);  
    this->hostMatrix=std::move(B.hostMatrix);
    this->gpuWorkers=std::move(B.gpuWorkers);
    return *this;
}

template <class Toperation>
MatrixMain<Toperation>& MatrixMain<Toperation>::operator+=(const Toperation& constantAddition)
{
    if(this->getColumnsReal()!=this->getRowsReal())
    {
        throw std::invalid_argument("La operación no se trata de una matriz cuadrada.");
    }
    if(!isDistributed)
    {
        distributeMatrixMySelfIntoGpus();
    }
    //Realizar la operación
    OperationType opType= ncclMultEnv->getOperationType();
    int i,j,numberOfDiagonalElements,matrixLocalIndex,idPhysicGpu,firtsBlockDiagonalPosition;
    Toperation* constantAdditionGpu;
    std::vector<Toperation*> constantAdditionGpus;
    //Pasar la constante a la gpu para poder operar
    for(i=0;i<ncclMultEnv->getGpuSizeOperationSystem();i++)
    {
        idPhysicGpu=gpuWorkers[i]->getGpuRankSystem();
        CUDACHECK(cudaSetDevice(idPhysicGpu));
        CUDACHECK(cudaMalloc((void**)&constantAdditionGpu,sizeof(Toperation)));
        CUDACHECK(cudaMemcpy(constantAdditionGpu,&constantAddition,sizeof(Toperation),cudaMemcpyHostToDevice));
        constantAdditionGpus.push_back(constantAdditionGpu);
    }
    //Operar
    for(i=0;i<ncclMultEnv->getGpuSizeOperationWorld()&&i<numberOfTotalBlocks;i++)
    {
        idPhysicGpu=gpuWorkers[i]->getGpuRankSystem();
        CUDACHECK(cudaSetDevice(idPhysicGpu));
        for(j=i,matrixLocalIndex=0;j<numberOfTotalBlocks;j+=ncclMultEnv->getGpuSizeOperationWorld(),matrixLocalIndex++)
        {
            if(std::get<0>(blocksInitialPositionDiagonal[j])!=-1)
            {
                firtsBlockDiagonalPosition=std::get<1>(blocksInitialPositionDiagonal[j]);
                numberOfDiagonalElements = min(std::get<2>(blocksInitialPositionDiagonal[j]),calculateBlockDimensionToCopy(calculateColumnColor(i), numberOfColumnBlocks, blockColumnSize, columnsUsed, columnsReal));
                MatrixUtilitiesCuda<Toperation>::axpyCublas(ncclMultEnv->getCublasHandlers()[idPhysicGpu],opType,numberOfDiagonalElements ,constantAdditionGpus[idPhysicGpu],&gpuWorkers[i]->getMatrixLocal(matrixLocalIndex)[firtsBlockDiagonalPosition],1,0,blockRowSize+1);
            }
        }
    }
    ncclMultEnv->waitAllCublasStreams();
    //Liberar recursos
    for(i=0;i<ncclMultEnv->getGpuSizeOperationSystem();i++)
    {
        idPhysicGpu=gpuWorkers[i]->getGpuRankSystem();
        CUDACHECK(cudaFree(constantAdditionGpus[i]));
    }
    setIsMatrixHostHere(false);
    return *this;
}

template <class Toperation>
MatrixMain<Toperation> MatrixMain<Toperation>::operator+(const Toperation& constantAddition)
{
    MatrixMain<Toperation> aux =*this;
    aux+=constantAddition;
    return aux;
}

template <class Toperation>
MatrixMain<Toperation>& MatrixMain<Toperation>::operator-=(const Toperation& constantSubstraction)
{
    *this+=-constantSubstraction;
    return *this;
}

template <class Toperation>
MatrixMain<Toperation> MatrixMain<Toperation>::operator-(const Toperation& constantSubstraction)
{
    MatrixMain<Toperation> aux =*this;
    aux-=constantSubstraction;
    return aux;
}

template <class Toperation>
MatrixMain<Toperation>& MatrixMain<Toperation>::operator+=(MatrixMain<Toperation>& maMain)
{
    axpy(1,maMain,*this);
    return *this;
}

template <class Toperation>
MatrixMain<Toperation> MatrixMain<Toperation>::operator+(MatrixMain<Toperation>& maMain)
{
    MatrixMain<Toperation> aux =*this;
    axpy(1,maMain,aux);
    return aux;
}

template <class Toperation>
MatrixMain<Toperation>& MatrixMain<Toperation>::operator-=(MatrixMain<Toperation>& maMain)
{
    axpy(-1,maMain,*this);
    return *this;
}

template <class Toperation>
MatrixMain<Toperation> MatrixMain<Toperation>::operator-(MatrixMain<Toperation>& maMain)
{
    MatrixMain<Toperation> aux =*this;
    axpy(-1,maMain,aux);
    return aux;
}

template <class Toperation>
MatrixMain<Toperation> MatrixMain<Toperation>::operator-()
{
    MatrixMain<Toperation> aux =*this;
    aux*=-1;
    return aux;
}

template<class Toperation>
MatrixMain<Toperation> operator+(const double& constantAddition, const MatrixMain<Toperation>& maMain)
{
    MatrixMain<Toperation> aux =maMain;
    aux+=constantAddition;
    return aux;
}

template<class Toperation>
MatrixMain<Toperation> operator*(const double& alpha, const MatrixMain<Toperation>& maMain)
{
    MatrixMain<Toperation> aux =maMain;
    aux*=alpha;
    return aux;
}

template<class Toperation>
MatrixMain<Toperation> operator-(const double& constantSubstraction, const MatrixMain<Toperation>& maMain)
{
    MatrixMain<Toperation> aux =maMain;
    aux-=constantSubstraction;
    aux*=-1;
    return aux;
}

template MatrixMain<double> operator+(const double& constantAddition, const MatrixMain<double>& maMain);
template MatrixMain<double> operator*(const double& alpha, const MatrixMain<double>& maMain);
template MatrixMain<double> operator-(const double& constantSubstraction, const MatrixMain<double>& maMain);
template MatrixMain<float> operator+(const double& constantAddition, const MatrixMain<float>& maMain);
template MatrixMain<float> operator*(const double& alpha, const MatrixMain<float>& maMain);
template MatrixMain<float> operator-(const double& constantSubstraction, const MatrixMain<float>& maMain);




template class MatrixMain<double>;
template class MatrixMain<float>;