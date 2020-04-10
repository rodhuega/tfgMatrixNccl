#include "NcclMultiplicationEnvironment.cuh"


template <class Toperation>
NcclMultiplicationEnvironment<Toperation>::NcclMultiplicationEnvironment(int gpuSizeWorld,int gpuRoot,OperationType opType,bool printMatrix)
{
    this->gpuSizeOperationWorld=-1;
    this->gpuRoot=gpuRoot;
    this->printMatrix=printMatrix;
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

    //Crear un cublasHandler con su stream correspondiente por cada gpu física del sistema
    int i;
    for(i=0;i<gpuSizeSystem;i++)
    {
        CUDACHECK(cudaSetDevice(i));
        cudaStream_t *newStream = new cudaStream_t;
        cublasHandle_t *newHandle = new cublasHandle_t;
        CUDACHECK(cudaStreamCreate(newStream));
        cublasStreams.push_back(newStream);
        CUBLASCHECK(cublasCreate(newHandle));
        CUBLASCHECK(cublasSetStream(*newHandle,*newStream));
        cublasHandlers.push_back(newHandle);
    }
    
    //SOLO DEBUG
    if(printMatrix)
    {
        std::cout<<"System gpus: "<<this->gpuSizeSystem<<". World gpus: "<<this->gpuSizeWorld<<std::endl;
    }

}

template <class Toperation>
NcclMultiplicationEnvironment<Toperation>::~NcclMultiplicationEnvironment()
{
    int i;
    for (i=0;i<cublasStreams.size();i++)
    {
        CUDACHECK(cudaSetDevice(i));
        CUBLASCHECK(cublasDestroy(*cublasHandlers[i]));
        CUDACHECK(cudaStreamDestroy(*cublasStreams[i]));
        if(i<gpuSizeOperationSystem)
        {
            NCCLCHECK(ncclCommDestroy(commOperation[i]));
        }
    }
    cublasStreams.clear();
    cublasHandlers.clear();
}

template <class Toperation>
int NcclMultiplicationEnvironment<Toperation>::getGpuSizeOperationWorld()
{
    return gpuSizeOperationWorld;
}
template <class Toperation>
int NcclMultiplicationEnvironment<Toperation>::getGpuSizeOperationSystem()
{
    return gpuSizeOperationSystem;
}
template <class Toperation>
int NcclMultiplicationEnvironment<Toperation>::getGpuSizeSystem()
{
    return gpuSizeSystem;
}
template <class Toperation>
int NcclMultiplicationEnvironment<Toperation>::getGpuSizeWorld()
{
    return gpuSizeWorld;
}
template <class Toperation>
int NcclMultiplicationEnvironment<Toperation>::getGpuRoot()
{
    return gpuRoot;
}
template <class Toperation>
void NcclMultiplicationEnvironment<Toperation>::waitAllCublasStreams()
{
    int i;
    for(i=0;i<cublasStreams.size();i++)
    {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(*cublasStreams[i]));
    }
}

template <class Toperation>
void NcclMultiplicationEnvironment<Toperation>::setOrAddMatrixMain(std::string id, MatrixMain<Toperation> *matrixMainGlobal)
{
    matricesMatrixMain[id] = matrixMainGlobal;
}

template <class Toperation>
MatrixMain<Toperation> *NcclMultiplicationEnvironment<Toperation>::getMainMatrix(std::string id)
{
    auto it = matricesMatrixMain.find(id);
    if (it == matricesMatrixMain.end())
    {
        throw std::invalid_argument("La matriz global existe");
    }
    return it->second;
}

template <class Toperation>
void NcclMultiplicationEnvironment<Toperation>::removeMatrixMain(std::string id,bool freeMemory)
{
    MatrixMain<Toperation> * auxMatrix=matricesMatrixMain[id];
    matricesMatrixMain.erase(id);
    if(freeMemory)
    {
        delete auxMatrix;
    }
}

template <class Toperation>
void NcclMultiplicationEnvironment<Toperation>::createNcclCommunicator(std::vector<CommSummaElement*> &commElements,std::set<int> &dimensionLogicDevices,bool setRowColor)
{
    int i,rank,gpuIdPhysical,logicRankIndex=0;
    cudaStream_t* newStream;
    //Vector que contiene los rangos de las gpus que acompañaran a esa gpu en el comunicador
    std::vector<int> logicRanks(gpuSizeOperationWorld);
    //Vector que contiene las gpus físicas que formaran parte del comunicador
    std::vector<int> devicesOfComm;
    std::vector<int> logicDevices;
    //Vector que en la posicion i tiene todas las gpus lógicas asociadas a esa física
    std::vector<std::vector<int>> physicalToLogic(gpuSizeOperationSystem);
    for(int gpuIdLogic: dimensionLogicDevices)
    {
        gpuIdPhysical=commElements[gpuIdLogic]->getIdPhysical();
        if (std::find(devicesOfComm.begin(), devicesOfComm.end(),gpuIdPhysical ) == devicesOfComm.end()) {
            devicesOfComm.push_back(gpuIdPhysical);
        }
        physicalToLogic[gpuIdPhysical].push_back(gpuIdLogic);
        logicDevices.push_back(gpuIdLogic);
        logicRanks[gpuIdLogic]=logicRankIndex;
        logicRankIndex++;
    }
    //Creación del comunicador y asignacion al elemento correspondiente del vector commElements
    ncclComm_t* newComm= new ncclComm_t[devicesOfComm.size()];    
    NCCLCHECK(ncclCommInitAll(newComm, devicesOfComm.size(), &devicesOfComm[0]));
    for(i=0;i<devicesOfComm.size();i++)
    {
        cudaSetDevice(devicesOfComm[i]);
        ncclCommUserRank(newComm[i],&rank);
        newStream= new cudaStream_t;
        CUDACHECK(cudaStreamCreate(newStream));
        for(int gpuIdLogic:physicalToLogic[devicesOfComm[i]])
        {
            if(setRowColor)
            {
                commElements[gpuIdLogic]->setRankCommRowPhysical(rank);
                commElements[gpuIdLogic]->setRankCommRowLogic(logicRanks[gpuIdLogic]);
                commElements[gpuIdLogic]->setCommRow(newComm[i]);
                commElements[gpuIdLogic]->setRowDevices(logicDevices);
                commElements[gpuIdLogic]->setStreamRow(newStream);
            }else
            {
                commElements[gpuIdLogic]->setRankCommColumnPhysical(rank);
                commElements[gpuIdLogic]->setRankCommColumnLogic(logicRanks[gpuIdLogic]);
                commElements[gpuIdLogic]->setCommColumn(newComm[i]);
                commElements[gpuIdLogic]->setColumnDevices(logicDevices);
                commElements[gpuIdLogic]->setStreamColumn(newStream);
            }
        }
    }
}

template <class Toperation>
void NcclMultiplicationEnvironment<Toperation>::setCommOperation(int gpuOperationSize)
{
    //Caso de que no se puede reutilizar el comunicador
    if(this->gpuSizeOperationWorld!=gpuOperationSize)
    {
        int i;
        if(this->gpuSizeOperationWorld!=-1)
        {
            for (i = 0; i < gpuSizeOperationSystem; ++i)
	        {
                NCCLCHECK(ncclCommDestroy(commOperation[i]));
	        }
        }
        //Creación del nuevo comunicador
        this->gpuSizeOperationWorld=gpuOperationSize;
        gpuSizeOperationSystem=min(gpuSizeOperationWorld,gpuSizeSystem);
        
        int arrayGpuSystemCommOperation[gpuSizeOperationSystem];
        for(i=0;i<gpuSizeOperationSystem;i++)
        {
            arrayGpuSystemCommOperation[i]=i;
        }
        commOperation=new ncclComm_t[gpuSizeOperationSystem];
        NCCLCHECK(ncclCommInitAll(commOperation, gpuSizeOperationSystem, arrayGpuSystemCommOperation));
    }
}

template <class Toperation>
std::string NcclMultiplicationEnvironment<Toperation>::generateRandomCandiateId()
{
    std::string str("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");

    std::random_device rd;
    std::mt19937 generator(rd());

    std::shuffle(str.begin(), str.end(), generator);

    return str.substr(0, 8);
}

template <class Toperation>
std::string NcclMultiplicationEnvironment<Toperation>::generateRandomId()
{
    bool randomGenerated=false;
    std::string id;
    while(!randomGenerated)
    {
        id=generateRandomCandiateId();
        auto it = matricesMatrixMain.find(id);
        if (it == matricesMatrixMain.end())
        {
            randomGenerated=true;
        }
    }
    return id;
}

template <class Toperation>
MatrixMain<Toperation> *NcclMultiplicationEnvironment<Toperation>::performCalculations(std::string idA,std::string idB, std::string idC)
{
    OperationProperties op;
    MatrixMain<Toperation> *ma, *mb, *mc;
    ma=getMainMatrix(idA);
    mb=getMainMatrix(idB);

    if(!MatrixUtilities<Toperation>::canMultiply(ma->getColumnsReal(),mb->getRowsReal()))
    {
        throw std::invalid_argument("La operacion no se puede realizar porque las columnas no coinciden con las filas. Columnas: " +std::to_string(ma->getColumnsReal())+ ", Filas: "+ std::to_string(mb->getRowsReal()));
    }

    if(!ma->getIsDistributed() && !mb->getIsDistributed())
    {
        op = MatrixUtilities<double>::getMeshAndMatrixSize(ma->getRowsReal(), ma->getColumnsReal(), mb->getRowsReal(), mb->getColumnsReal(), gpuSizeWorld);
        
        ma->setRowsUsed(op.rowsA);
        ma->setColumnsUsed(op.columnsAorRowsB);
        mb->setRowsUsed(op.columnsAorRowsB);
        mb->setColumnsUsed(op.columnsB);

        if (printMatrix)
        {
            std::cout << "NGpus: " << op.gpuSize << ", meshRowSize: " << op.meshRowSize << ", meshColumnSize: " << op.meshColumnSize << ", blockRowSizeA: " << \
            op.blockRowSizeA << ", blockColumnSizeA: " << op.blockColumnSizeA << ", blockRowSizeB: " << op.blockRowSizeB << ", blockColumnSizeB: " << \
            op.blockColumnSizeB << ", rowsA: " << op.rowsA << ", columnsAorRowsB: " << op.columnsAorRowsB << ", columnsB: " << op.columnsB << std::endl;

            std::cout << "A-> Rows: " << ma->getRowsReal() << ", Columns: " << ma->getColumnsReal() << ", Matriz A:" << std::endl;
            MatrixUtilities<Toperation>::printMatrix(ma->getRowsReal(), ma->getColumnsReal(), ma->getHostMatrix());
            std::cout << "B-> Rows: " << mb->getRowsReal() << ", Columns: " << mb->getColumnsReal() << ", Matriz B:" << std::endl;
            MatrixUtilities<Toperation>::printMatrix(mb->getRowsReal(), mb->getColumnsReal(), mb->getHostMatrix());
        }
        
        setCommOperation(op.gpuSize);
        ma->setMatrixOperationProperties(op.meshRowSize,op.meshColumnSize,op.blockRowSizeA,op.blockColumnSizeA);
        mb->setMatrixOperationProperties(op.meshRowSize,op.meshColumnSize,op.blockRowSizeB,op.blockColumnSizeB);
        ma->distributeMatrixIntoGpus();
        mb->distributeMatrixIntoGpus();
        ma->waitAllStreamsOfAllWorkers();
        mb->waitAllStreamsOfAllWorkers();

        // MatrixUtilitiesCuda<Toperation>::cudaDebugMatricesLocalDifferentGpuWorkers(gpuSizeOperationWorld,ma->getBlockRowSize(),ma->getBlockColumnSize(),ma->getGpuWorkers());

        mc=ncclSumma(ma,mb,op.meshRowSize,op.meshColumnSize);
        if(idC!="")
        {
            mc->setId(idC);
        }
        // MatrixUtilitiesCuda<Toperation>::cudaDebugMatricesLocalDifferentGpuWorkers(gpuSizeOperationWorld,gpuSizeSystem,mc->getBlockRowSize(),mc->getBlockColumnSize(),mc->getGpuWorkers());
    }
    return mc;
}

template <class Toperation>
MatrixMain<Toperation>*  NcclMultiplicationEnvironment<Toperation>::ncclSumma(MatrixMain<Toperation>* matrixA, MatrixMain<Toperation>* matrixB, int meshRowsSize, int meshColumnsSize)
{
    int i,j,gpuRank,rowColor,columnColor;
    int rowsA = matrixA->getRowsUsed();
    int columnsAorRowsB = matrixA->getColumnsUsed();
    int columnsB = matrixB->getColumnsUsed();
    int blockSizeA = matrixA->getBlockSize();
    int blockSizeB = matrixB->getBlockSize();
    int blockRowSizeA = matrixA->getBlockRowSize();
    int blockColumnsSizeA = matrixA->getBlockColumnSize();
    int blockColumnsSizeB = matrixB->getBlockColumnSize();
    int blockRowSizeB = matrixB->getBlockRowSize();
    //Creación del esquelo del elemento que va a ser devuelto
    MatrixMain<Toperation> *mc= new MatrixMain<Toperation>(this,generateRandomId(),matrixA->getRowsReal(),matrixB->getColumnsReal());
    mc->setIsDistributed(true);
    mc->setRowsUsed(matrixA->getRowsUsed());
    mc->setColumnsUsed(matrixB->getColumnsUsed());
    mc->setMatrixOperationProperties(meshRowsSize,meshColumnsSize,blockRowSizeA,blockColumnsSizeB);

    //Reserva de las matrices buffer para cada gpu
    std::vector<Toperation*> gpuAuxiliarMatricesA,gpuAuxiliarMatricesB;
    //Sets en el que cada elemeneto es el color y tienen un vector de las ids lógica de los elementos que pertenecen a ese color. Usado dentro del bucle de Summa porque find O(log n)
    std::vector<std::set<int>> rowColorsLogic(meshRowsSize),columnColorsLogic(meshColumnsSize);
    //Array de vecotores que tendra los comunicadores(array de ncclComm_t) de cada gpu lógica
    std::vector<CommSummaElement*> commElements(gpuSizeOperationWorld);
    //Inicialización de los vectores descritos previamente para tener la información necesaria para realizar Summa
    for(i=0;i<gpuSizeOperationWorld;i++)
    {
        int gpuRealId=MatrixUtilitiesCuda<Toperation>::getRealGpuId(i,gpuSizeSystem);
        CUDACHECK(cudaSetDevice(gpuRealId));
        Toperation *gpuAuxA=MatrixUtilitiesCuda<Toperation>::cudaMatrixMemoryAllocation(blockRowSizeA,blockColumnsSizeA,cublasStreams[gpuRealId]);
        Toperation *gpuAuxB=MatrixUtilitiesCuda<Toperation>::cudaMatrixMemoryAllocation(blockRowSizeB,blockColumnsSizeB,cublasStreams[gpuRealId]);
        gpuAuxiliarMatricesA.push_back(gpuAuxA);gpuAuxiliarMatricesB.push_back(gpuAuxB);

        rowColor=matrixA->calculateRowColor(i);
        columnColor=matrixA->calculateColumnColor(i);
        rowColorsLogic[rowColor].insert(i);
        columnColorsLogic[columnColor].insert(i);
        commElements[i]=new CommSummaElement(i,gpuRealId,rowColor,columnColor);
    }
    //Creacion de los comunicadores
    std::set<int> rowsColorSet,columnColorSet;
    for(i=0;i<meshRowsSize||i<meshColumnsSize;i++)
    {
        if(i<meshRowsSize)
        {
            rowsColorSet = rowColorsLogic[i];
            createNcclCommunicator(commElements,rowsColorSet,true);

        }
        if(i<meshColumnsSize)
        {
            columnColorSet = columnColorsLogic[i];
            createNcclCommunicator(commElements,columnColorSet,false);
        }
    }

    //Realizacion de las operaciones matematicas. Algoritmo Summa
    for (i = 0; i < meshColumnsSize; i++)
    {
        //Copiar las matrices que tocan al buffer
        for(gpuRank=0;gpuRank<gpuSizeOperationWorld;gpuRank++)
        {
            int gpuRealId=MatrixUtilitiesCuda<Toperation>::getRealGpuId(gpuRank,gpuSizeSystem);
            CUDACHECK(cudaSetDevice(gpuRealId));
            if (columnColorsLogic[(i % meshColumnsSize)].find(gpuRank)!=columnColorsLogic[(i % meshColumnsSize)].end())
            {
                CUDACHECK(cudaMemcpyAsync(gpuAuxiliarMatricesA[gpuRank],matrixA->getGpuWorkers()[gpuRank]->getMatrixLocal(i / meshColumnsSize),blockSizeA*sizeof(Toperation),cudaMemcpyDeviceToDevice,*matrixA->getGpuWorkers()[gpuRank]->getStream(i / meshColumnsSize)));
            }
            if (rowColorsLogic[(i % meshRowsSize)].find(gpuRank)!=rowColorsLogic[(i % meshRowsSize)].end())
            {
                CUDACHECK(cudaMemcpyAsync(gpuAuxiliarMatricesB[gpuRank],matrixB->getGpuWorkers()[gpuRank]->getMatrixLocal(i / meshColumnsSize),blockSizeB*sizeof(Toperation),cudaMemcpyDeviceToDevice,*matrixB->getGpuWorkers()[gpuRank]->getStream(i / meshColumnsSize)));
            }
        }
        //Esperar esa copia
        matrixA->waitAllStreamsOfAllWorkers();
        matrixB->waitAllStreamsOfAllWorkers();
        
        //Realizacion de las comunicaciones
        std::vector<std::vector<cudaStream_t*>> commStreams(gpuSizeOperationWorld);
        NCCLCHECK(ncclGroupStart());
        for(gpuRank=0;gpuRank<gpuSizeOperationWorld;gpuRank++)
	    {
            if(commElements[gpuRank]->getRankCommRowLogic()==(i % meshRowsSize))
            {
                for(int gpuRankComm:commElements[gpuRank]->getRowDevices())
                {
                    CUDACHECK(cudaSetDevice(MatrixUtilitiesCuda<Toperation>::getRealGpuId(gpuRankComm,gpuSizeSystem)));
                    NCCLCHECK(ncclBroadcast(gpuAuxiliarMatricesA[gpuRank],gpuAuxiliarMatricesA[gpuRankComm],blockSizeA,
                        basicOperationType,commElements[gpuRank]->getRankCommRowPhysical(),commElements[gpuRankComm]->getCommRow(),
                        *commElements[gpuRankComm]->getStreamRow()));
                }
            }
            if(commElements[gpuRank]->getRankCommColumnLogic()==(i % meshColumnsSize))
            {
                for(int gpuRankComm:commElements[gpuRank]->getColumnDevices())
                {
                    int realId=MatrixUtilitiesCuda<Toperation>::getRealGpuId(gpuRankComm,gpuSizeSystem);
                    CUDACHECK(cudaSetDevice(realId));
                    NCCLCHECK(ncclBroadcast(gpuAuxiliarMatricesB[gpuRank],gpuAuxiliarMatricesB[gpuRankComm],blockSizeB,
                        basicOperationType,commElements[gpuRank]->getRankCommColumnPhysical(),commElements[gpuRankComm]->getCommColumn(),
                        *commElements[gpuRankComm]->getStreamColumn()));
                }
            }
        }
        NCCLCHECK(ncclGroupEnd());
        //Esperar las comunicaciones
        for(gpuRank=0;gpuRank<commElements.size();gpuRank++)
        {
            CUDACHECK(cudaSetDevice(MatrixUtilitiesCuda<Toperation>::getRealGpuId(gpuRank,gpuSizeSystem)));
            CUDACHECK(cudaStreamSynchronize(*commElements[gpuRank]->getStreamRow()));
            CUDACHECK(cudaStreamSynchronize(*commElements[gpuRank]->getStreamColumn()));
        }
        //Realización de todas las multiplicaciones
        for(gpuRank=0;gpuRank<gpuSizeOperationWorld;gpuRank++)
	    {
            // std::cout<<"Aux A: Iteracion: "<<i<<", gpuRank: "<<gpuRank<<std::endl;
            // MatrixUtilitiesCuda<Toperation>::cudaPrintOneMatrixCall(matrixA->getBlockRowSize(),matrixA->getBlockColumnSize(),gpuAuxiliarMatricesA[gpuRank]);
            // std::cout<<"Aux B: Iteracion: "<<i<<", gpuRank: "<<gpuRank<<std::endl;
            // MatrixUtilitiesCuda<Toperation>::cudaPrintOneMatrixCall(matrixB->getBlockRowSize(),matrixB->getBlockColumnSize(),gpuAuxiliarMatricesB[gpuRank]);
            int gpuRealId=MatrixUtilitiesCuda<Toperation>::getRealGpuId(gpuRank,gpuSizeSystem);
            CUDACHECK(cudaSetDevice(gpuRealId));
            MatrixUtilitiesCuda<Toperation>::matrixCublasMultiplication(cublasHandlers[gpuRealId],opType,blockRowSizeA,blockRowSizeB,blockColumnsSizeB,gpuAuxiliarMatricesA[gpuRank],gpuAuxiliarMatricesB[gpuRank],mc->getGpuWorkers()[gpuRank]->getMatrixLocal(0));
        }
        waitAllCublasStreams();
        // std::cout<<"Iteracion: "<<i<<std::endl;
        // MatrixUtilitiesCuda<Toperation>::cudaDebugMatricesLocalDifferentGpuWorkers(gpuSizeOperationWorld,mc->getBlockRowSize(),mc->getBlockColumnSize(),mc->getGpuWorkers());
    }
    //Liberar los recursos utilizados
    for(i=0;i<gpuSizeOperationWorld;i++)
    {
        delete commElements[i];
        CUDACHECK(cudaSetDevice(MatrixUtilitiesCuda<Toperation>::getRealGpuId(gpuRank,gpuSizeSystem)));
        MatrixUtilitiesCuda<Toperation>::matrixFree(gpuAuxiliarMatricesA[i]);
        MatrixUtilitiesCuda<Toperation>::matrixFree(gpuAuxiliarMatricesB[i]);
    }
    commElements.clear();
    return mc;
}


template class NcclMultiplicationEnvironment<double>;
template class NcclMultiplicationEnvironment<float>;

