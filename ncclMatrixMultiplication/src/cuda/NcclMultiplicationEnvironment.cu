#include "NcclMultiplicationEnvironment.cuh"


template <class Toperation>
NcclMultiplicationEnvironment<Toperation>::NcclMultiplicationEnvironment(int gpuSizeWorld,int gpuRoot,OperationType opType)
{
    this->gpuSizeOperationWorld=-1;
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

template <class Toperation>//////////////////Por revisar esta implementacion
MatrixMain<Toperation> *NcclMultiplicationEnvironment<Toperation>::getMainMatrix(std::string id, bool create)
{
    auto it = matricesMatrixMain.find(id);
    if (it == matricesMatrixMain.end())
    {
        if (!create)
        {
            throw std::invalid_argument("La matriz global existe");
        }
        else
        {
            // Toperation *matrixAux = getMatrixGlobalSimplePointer(id);
            // dimensions dimensionsAux = matricesGlobalDimensions[id];
            // MatrixMain<Toperation> *res = new MatrixMain<Toperation>(std::get<0>(dimensionsAux), std::get<1>(dimensionsAux));
            // setOrAddMatrixMain(id, res);
            // return res;
        }
    }
    return it->second;
}

template <class Toperation>
ncclComm_t* NcclMultiplicationEnvironment<Toperation>::createNcclCommunicator(std::vector<int> devicesOfComm)
{
    ncclComm_t* newComm= new ncclComm_t[devicesOfComm.size()];
    NCCLCHECK(ncclCommInitAll(newComm, devicesOfComm.size(), &devicesOfComm[0]));
    return newComm;
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
std::vector<int> NcclMultiplicationEnvironment<Toperation>::convertSetToVector(std::set<int> &s)
{
    std::vector<int> v(s.size());
    std::copy(s.begin(), s.end(), v.begin());
    return v;
}

template <class Toperation>
void NcclMultiplicationEnvironment<Toperation>::performCalculations(std::string idA,std::string idB, std::string idC,bool printMatrix)
{
    OperationProperties op;
    MatrixMain<Toperation> *ma, *mb, *mc;
    ma=getMainMatrix(idA,false);
    mb=getMainMatrix(idB,false);

    if(!MatrixUtilities<Toperation>::canMultiply(ma->getColumnsReal(),mb->getRowsReal()))
    {
        throw std::invalid_argument("La operacion no se puede realizar porque las columnas no coinciden con las filas. Columnas: " +std::to_string(ma->getColumnsReal())+ ", Filas: "+ std::to_string(mb->getRowsReal()));
    }

    if(!ma->getIsDistributed() && !mb->getIsDistributed())
    {
        op = MatrixUtilities<double>::getMeshAndMatrixSize(ma->getRowsReal(), ma->getColumnsReal(), mb->getRowsReal(), mb->getColumnsReal(), gpuSizeWorld);
        std::cout << "NGpus: " << op.gpuSize << ", meshRowSize: " << op.meshRowSize << ", meshColumnSize: " << op.meshColumnSize << ", blockRowSizeA: " << \
            op.blockRowSizeA << ", blockColumnSizeA: " << op.blockColumnSizeA << ", blockRowSizeB: " << op.blockRowSizeB << ", blockColumnSizeB: " << \
            op.blockColumnSizeB << ", rowsA: " << op.rowsA << ", columnsAorRowsB: " << op.columnsAorRowsB << ", columnsB: " << op.columnsB << std::endl;
        
        ma->setRowsUsed(op.rowsA);
        ma->setColumnsUsed(op.columnsAorRowsB);
        
        mb->setRowsUsed(op.columnsAorRowsB);
        mb->setColumnsUsed(op.columnsB);

        if (printMatrix)
        {
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

        mc=mpiSumma(ma,mb,op.meshRowSize,op.meshColumnSize);
        //FALTA ASIGNAR LA ID
        MatrixUtilitiesCuda<Toperation>::cudaDebugMatricesLocalDifferentGpuWorkers(gpuSizeOperationWorld,mc->getBlockRowSize(),mc->getBlockColumnSize(),mc->getGpuWorkers());
    }
}

template <class Toperation>
MatrixMain<Toperation>*  NcclMultiplicationEnvironment<Toperation>::mpiSumma(MatrixMain<Toperation>* matrixA, MatrixMain<Toperation>* matrixB, int meshRowsSize, int meshColumnsSize)
{
    int i,gpuRank;
    int rowsA = matrixA->getRowsUsed();
    int columnsAorRowsB = matrixA->getColumnsUsed();
    int columnsB = matrixB->getColumnsUsed();
    int blockSizeA = matrixA->getBlockSize();
    int blockSizeB = matrixB->getBlockSize();
    int blockRowSizeA = matrixA->getBlockRowSize();
    int blockColumnsSizeA = matrixA->getBlockColumnSize();
    int blockColumnsSizeB = matrixB->getBlockColumnSize();
    int blockRowSizeB = matrixB->getBlockRowSize();
    //Creacion del esquelo del elemento que va a ser devuelto
    MatrixMain<Toperation> *mc= new MatrixMain<Toperation>(this,generateRandomId(),matrixA->getRowsReal(),matrixB->getColumnsReal());
    mc->setIsDistributed(true);
    mc->setRowsUsed(matrixA->getRowsUsed());
    mc->setColumnsUsed(matrixB->getColumnsUsed());
    mc->setMatrixOperationProperties(meshRowsSize,meshColumnsSize,blockRowSizeA,blockColumnsSizeB);
    //Reserva de las matrices buffer para cada gpu y conseguir a que columna y fila pertenece cada gpu. Posicion i de los vectores asociadas a esa i de gpuWorker
    std::vector<Toperation*> gpuAuxiliarMatricesA,gpuAuxiliarMatricesB;
    //Sets que para cada elemento que indica el color tienen un vector de la id logica de los elementos que pertenecen a ese color
    std::vector<std::set<int>> rowColorsLogic(meshRowsSize),columnColorsLogic(meshColumnsSize);
    //Sets para crear los comunicadores con la gpu fisica
    std::vector<std::set<int>> rowColorPhysicalSet(meshRowsSize),columnColorPhysicalSet(meshColumnsSize);
    //Array de vecotores que tendra los comunicadores(array de ncclComm_t) de cada gpu logica
    std::vector<ncclComm_t*> rowColorComm(meshRowsSize),columColorComm(meshColumnsSize);
    for(i=0;i<gpuSizeOperationWorld;i++)
    {
        int gpuRealId=MatrixUtilitiesCuda<Toperation>::getRealGpuId(i,gpuSizeSystem);
        CUDACHECK(cudaSetDevice(gpuRealId));
        Toperation *gpuAuxA=MatrixUtilitiesCuda<Toperation>::cudaMatrixMemoryAllocation(blockRowSizeA,blockColumnsSizeA,cublasStreams[gpuRealId]);
        Toperation *gpuAuxB=MatrixUtilitiesCuda<Toperation>::cudaMatrixMemoryAllocation(blockRowSizeB,blockColumnsSizeB,cublasStreams[gpuRealId]);
        gpuAuxiliarMatricesA.push_back(gpuAuxA);gpuAuxiliarMatricesB.push_back(gpuAuxB);
        rowColorsLogic[matrixA->calculateRowColor(i)].insert(i);
        columnColorsLogic[matrixA->calculateColumnColor(i)].insert(i);
        rowColorPhysicalSet[matrixA->calculateRowColor(i)].insert(gpuRealId);
        columnColorPhysicalSet[matrixA->calculateColumnColor(i)].insert(gpuRealId);
    }
    //Creacion de los comunicadores
    std::vector<int> auxVec;
    for(i=0;i<meshRowsSize||i<meshColumnsSize;i++)
    {
        if(i<meshRowsSize)
        {
            auxVec=convertSetToVector(rowColorPhysicalSet[i]);
            rowColorComm[i]=createNcclCommunicator(auxVec);

        }
        if(i<meshColumnsSize)
        {
            auxVec=convertSetToVector(columnColorPhysicalSet[i]);
            columColorComm[i]=createNcclCommunicator(auxVec);
        }
    }

    //Realizacion de las operaciones matematicas
    for (i = 0; i < meshColumnsSize; i++)
    {
        //Copiar las matrices que tocan al buffer
        for(gpuRank=0;gpuRank<gpuSizeOperationWorld;gpuRank++)
        {
            int gpuRealId=MatrixUtilitiesCuda<Toperation>::getRealGpuId(gpuRank,gpuSizeSystem);
            CUDACHECK(cudaSetDevice(gpuRealId));
            if (columnColorsLogic[(i % meshColumnsSize)].find(gpuRank)!=columnColorsLogic[(i % meshColumnsSize)].end())
            {
                // memcpy(matrixAuxiliarA, matrixLocalA.getMatrixLocal(i / meshColumnsSize), blockSizeA * sizeof(Toperation));
                CUDACHECK(cudaMemcpyAsync(gpuAuxiliarMatricesA[gpuRank],matrixA->getGpuWorkers()[gpuRank]->getMatrixLocal(i / meshColumnsSize),blockSizeA*sizeof(Toperation),cudaMemcpyDeviceToDevice,*matrixA->getGpuWorkers()[gpuRank]->getStream(i / meshColumnsSize)));
            }
            if (rowColorsLogic[(i % meshRowsSize)].find(gpuRank)!=rowColorsLogic[(i % meshRowsSize)].end())
            {
                CUDACHECK(cudaMemcpyAsync(gpuAuxiliarMatricesB[gpuRank],matrixB->getGpuWorkers()[gpuRank]->getMatrixLocal(i / meshColumnsSize),blockSizeB*sizeof(Toperation),cudaMemcpyDeviceToDevice,*matrixB->getGpuWorkers()[gpuRank]->getStream(i / meshColumnsSize)));
                // memcpy(matrixAuxiliarB, matrixLocalB.getMatrixLocal(i / meshRowsSize), blockSizeB * sizeof(Toperation));
            }
        }
        //Esperar esa copia
        matrixA->waitAllStreamsOfAllWorkers();
        matrixB->waitAllStreamsOfAllWorkers();
        //Realizacion de las comunicaciones
        NCCLCHECK(ncclGroupStart());
        for(gpuRank=0;gpuRank<gpuSizeOperationWorld;gpuRank++)
	    {
            int gpuRealIdCommRowRoot=MatrixUtilitiesCuda<Toperation>::getRealGpuId((i % meshColumnsSize),gpuSizeSystem);
            int gpuRealIdCommColumnRoot=MatrixUtilitiesCuda<Toperation>::getRealGpuId((i % meshRowsSize),gpuSizeSystem);
            NCCLCHECK(ncclBroadcast(gpuAuxiliarMatricesA[gpuRank],gpuAuxiliarMatricesA[gpuRank],blockSizeA,
                basicOperationType,gpuRealIdCommRowRoot,*rowColorComm[matrixA->getGpuWorkers()[gpuRank]->getGpuRankSystem()],
                *matrixA->getGpuWorkers()[gpuRank]->getStream(i / meshColumnsSize)));
            NCCLCHECK(ncclBroadcast(gpuAuxiliarMatricesB[gpuRank],gpuAuxiliarMatricesB[gpuRank],blockSizeA,
                basicOperationType,gpuRealIdCommColumnRoot,*rowColorComm[matrixA->getGpuWorkers()[gpuRank]->getGpuRankSystem()],
                *matrixA->getGpuWorkers()[gpuRank]->getStream(i / meshColumnsSize)));
        }
        NCCLCHECK(ncclGroupEnd());
        matrixA->waitAllStreamsOfAllWorkers();
        matrixB->waitAllStreamsOfAllWorkers();
        for(gpuRank=0;gpuRank<gpuSizeOperationWorld;gpuRank++)
	    {
            int gpuRealId=MatrixUtilitiesCuda<Toperation>::getRealGpuId(gpuRank,gpuSizeSystem);
            MatrixUtilitiesCuda<Toperation>::matrixCublasMultiplication(cublasHandlers[gpuRealId],blockRowSizeA,blockRowSizeB,blockColumnsSizeB,gpuAuxiliarMatricesA[gpuRank],gpuAuxiliarMatricesB[gpuRank],mc->getGpuWorkers()[gpuRank]->getMatrixLocal(0));
        }
        waitAllCublasStreams();

        // if (columnColor == (i % meshColumnsSize))
        // {
        //     memcpy(matrixAuxiliarA, matrixLocalA.getMatrixLocal(i / meshColumnsSize), blockSizeA * sizeof(Toperation));
        // }
        // if (rowColor == (i % meshRowsSize))
        // {
        //     memcpy(matrixAuxiliarB, matrixLocalB.getMatrixLocal(i / meshRowsSize), blockSizeB * sizeof(Toperation));
        // }
        // MPI_Bcast(matrixAuxiliarA, blockSizeA, basicOperationType, (i % meshColumnsSize), commRow);
        // MPI_Bcast(matrixAuxiliarB, blockSizeB, basicOperationType, (i % meshRowsSize), commCol);

        // MatrixUtilities<Toperation>::matrixBlasMultiplication(blockRowSizeA, blockRowSizeB, blockColumnsSizeB, matrixAuxiliarA, matrixAuxiliarB, matrixLocalC);
    }
    // //Liberacion de las matrices auxiliares que realizaban computo
    // MatrixUtilities<Toperation>::matrixFree(matrixAuxiliarA);
    // MatrixUtilities<Toperation>::matrixFree(matrixAuxiliarB);
    // return matrixLocalC;

    //FALTA LIBERAR MEMORIA
    return mc;
    
}


template class NcclMultiplicationEnvironment<double>;
