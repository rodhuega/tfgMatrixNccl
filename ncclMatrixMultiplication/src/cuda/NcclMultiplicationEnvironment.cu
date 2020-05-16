#include "NcclMultiplicationEnvironment.cuh"


template <class Toperation>
NcclMultiplicationEnvironment<Toperation>::NcclMultiplicationEnvironment(int gpuSizeWorld,int gpuRoot,OperationType opType,bool printMatrix)
{
    this->gpuSizeOperationWorld=-1;
    this->gpuRoot=gpuRoot;
    this->printMatrix=printMatrix;
    this->lastMeshRowSize=-1;
    this->lastMeshColumnSize=-1;
    this->lastBlockRowSizeA=-1;
    this->lastBlockColumnSizeA=-1;
    this->lastBlockRowSizeB=-1;
    this->lastBlockColumnSizeB=-1;
    CUDACHECK(cudaGetDeviceCount(&gpuSizeSystem));

    if(gpuSizeWorld!=-1)
    {
        this->gpuSizeWorld=max(gpuSizeWorld,4);
    }else
    {
        this->gpuSizeWorld=max(gpuSizeSystem,4);
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
    
    //Eliminar los elementos del map que contiene los comunicadores
    for(auto itMap=summaComms.begin();itMap!=summaComms.end();itMap++)
    {
        auto actualElement=itMap->second;
        std::vector<CommSummaElement*> commActual=std::get<0>(actualElement);
        for(i=0;i<commActual.size();i++)
        {
            delete commActual[i];
        }
        commActual.clear();
        std::get<1>(actualElement).clear();
        std::get<2>(actualElement).clear();
    }
    summaComms.clear();
    eraseBufferMatrix();
    for (i=0;i<cublasStreams.size();i++)
    {
        CUDACHECK(cudaSetDevice(i));
        CUBLASCHECK(cublasDestroy(*cublasHandlers[i]));
        CUDACHECK(cudaStreamDestroy(*cublasStreams[i]));
        delete cublasHandlers[i];
        delete cublasStreams[i];
        // CUDACHECK(cudaDeviceReset());
    }
    cublasStreams.clear();
    cublasHandlers.clear();
    
}

template <class Toperation>
ncclDataType_t NcclMultiplicationEnvironment<Toperation>::getBasicOperationType()
{
    return basicOperationType;
}

template <class Toperation>
void NcclMultiplicationEnvironment<Toperation>::eraseBufferMatrix()
{
    int i;
    for(i=0;i<gpuAuxiliarMatricesA.size() || i<gpuAuxiliarMatricesB.size();i++)
    {
        CUDACHECK(cudaSetDevice(MatrixUtilitiesCuda<Toperation>::getRealGpuId(i,gpuSizeSystem)));
        if(i<gpuAuxiliarMatricesA.size())
        {
            MatrixUtilitiesCuda<Toperation>::matrixFreeGPU(gpuAuxiliarMatricesA[i]);
        }
        if(i<gpuAuxiliarMatricesB.size())
        {
            MatrixUtilitiesCuda<Toperation>::matrixFreeGPU(gpuAuxiliarMatricesB[i]);
        }
    }
    gpuAuxiliarMatricesA.clear();
    gpuAuxiliarMatricesB.clear();
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
OperationType NcclMultiplicationEnvironment<Toperation>::getOperationType()
{
    return opType;
}

template <class Toperation>
std::vector<cublasHandle_t*> NcclMultiplicationEnvironment<Toperation>::getCublasHandlers()
{
    return cublasHandlers;
}

template <class Toperation>
void NcclMultiplicationEnvironment<Toperation>::setGpuSizeOperationWorld(int gpuSizeOperationWorld)
{
    this->gpuSizeOperationWorld=gpuSizeOperationWorld;
}
template <class Toperation>
void NcclMultiplicationEnvironment<Toperation>::setGpuSizeOperationSystem(int gpuSizeOperationSystem)
{
    this->gpuSizeOperationSystem=gpuSizeOperationSystem;
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
void NcclMultiplicationEnvironment<Toperation>::createNcclCommunicator(std::vector<CommSummaElement*> &commElements,std::set<int> &dimensionLogicDevices,bool setRowColor)
{
    int i,j,k,rank,gpuIdPhysical,logicRankIndex=0;
    //Vector que contiene los rangos de las gpus que acompañaran a esa gpu en el comunicador
    std::vector<int> logicRanks(gpuSizeOperationWorld);

    //Vector que contiene las gpus físicas que formaran parte del comunicador para cada grupo. Primer vector son las gpus físicas
    std::vector<std::vector<int>> devicesOfComm;
    std::vector<std::vector<int>> logicDevices;
    //Vector que en la posicion i tiene todas las gpus lógicas asociadas a esa física para cada grupo. Primer vector son las gpus físicas
    std::vector<std::vector<std::vector<int>>> physicalToLogic;
    bool assigned=false;
    for(int gpuIdLogic: dimensionLogicDevices)
    {
        gpuIdPhysical=commElements[gpuIdLogic]->getIdPhysical();
        for(i=0;i<devicesOfComm.size();i++)
        {   //Ver si se puede agregar como gpu física
            if (std::find(devicesOfComm[0].begin(), devicesOfComm[0].end(),gpuIdPhysical ) == devicesOfComm[0].end()) {
                devicesOfComm[0].push_back(gpuIdPhysical);
                assigned=true;
                physicalToLogic[0][gpuIdPhysical].push_back(gpuIdLogic);
                logicDevices[0].push_back(gpuIdLogic);
            }
        }

        if(!assigned)
        {//En caso de que no no haya gpu física para esa gpu lógica
            std::vector<int> newDevicesOfComm;newDevicesOfComm.push_back(gpuIdPhysical);
            std::vector<std::vector<int>> newPhysicalToLogic(gpuSizeOperationSystem);newPhysicalToLogic[gpuIdPhysical].push_back(gpuIdLogic);
            std::vector<int> newLogicDevices;newLogicDevices.push_back(gpuIdLogic);
            logicDevices.push_back(newLogicDevices);
            physicalToLogic.push_back(newPhysicalToLogic);
            devicesOfComm.push_back(newDevicesOfComm);
        }
        assigned=false;
        logicRanks[gpuIdLogic]=logicRankIndex;
        logicRankIndex++;
    }
    //Creación del comunicador y asignacion al elemento correspondiente del vector commElements
    for(j=0;j<devicesOfComm.size();j++)
    {
        ncclComm_t* newComm= new ncclComm_t[devicesOfComm[j].size()];    
        NCCLCHECK(ncclCommInitAll(newComm, devicesOfComm[j].size(), &devicesOfComm[j][0]));
        for(i=0;i<devicesOfComm[j].size();i++)
        {
            cudaSetDevice(devicesOfComm[j][i]);
            ncclCommUserRank(newComm[i],&rank);
            
            std::vector<int> auxDevicesOfComm=devicesOfComm[j];
            std::vector<std::vector<int>> auxPhysicalToLogic=physicalToLogic[j];
            for(int gpuIdLogic:auxPhysicalToLogic[auxDevicesOfComm[i]])
            {
                if(setRowColor)
                {
                    commElements[gpuIdLogic]->setRankCommRowLogic(logicRanks[gpuIdLogic]);
                    commElements[gpuIdLogic]->setRankCommRowPhysical(rank);
                    commElements[gpuIdLogic]->setCommRow(newComm[i]);
                    if(logicDevices[0][0]!=logicDevices[j][0])
                    {
                        for(k=0;k<logicDevices.size();k++)
                        {//En caso de que no no haya gpu física para esa gpu lógica
                            for(int ll=0;ll<logicDevices[k].size();ll++)
                            {
                                commElements[logicDevices[k][ll]]->addCommRowMySelf(newComm[i]);
                                commElements[gpuIdLogic]->addCommRowMySelf(newComm[i]);
                            }
                        }
                    }
                    commElements[gpuIdLogic]->setRowDevices(logicDevices);
                }else
                {
                    commElements[gpuIdLogic]->setRankCommColumnLogic(logicRanks[gpuIdLogic]);
                    commElements[gpuIdLogic]->setRankCommColumnPhysical(rank);
                    commElements[gpuIdLogic]->setCommColumn(newComm[i]);
                    if(logicDevices[0][0]!=logicDevices[j][0])
                    {
                        for(k=0;k<logicDevices.size();k++)
                        {//En caso de que no no haya gpu física para esa gpu lógica
                            for(int ll=0;ll<logicDevices[k].size();ll++)
                            {
                                commElements[logicDevices[k][ll]]->addCommColumnMySelf(newComm[i]);
                                commElements[gpuIdLogic]->addCommColumnMySelf(newComm[i]);
                            }
                        }
                    }
                    commElements[gpuIdLogic]->setColumnDevices(logicDevices);
                }
            }
        }
    }
}

template <class Toperation>
std::tuple<std::vector<CommSummaElement*>,std::vector<std::set<int>>,std::vector<std::set<int>>> NcclMultiplicationEnvironment<Toperation>::getOrCreateCommunicators(int meshRowSize, int meshColumnSize,MatrixMain<Toperation>* matrixA)
{
    auto itMapComm= summaComms.find(meshRowSize);
    if(itMapComm==summaComms.end())
    {
        //Sets en el que cada elemeneto es el color y tienen un vector de las ids lógica de los elementos que pertenecen a ese color. Usado dentro del bucle de Summa porque find O(log n)
        std::vector<std::set<int>> rowColorsLogic(meshRowSize),columnColorsLogic(meshColumnSize);
        //Array de vecotores que tendra los comunicadores(array de ncclComm_t) de cada gpu lógica
        std::vector<CommSummaElement*> commElements(gpuSizeOperationWorld);
        int i,rowColor,columnColor;
        //Inicialización de los vectores descritos previamente para tener la información necesaria para realizar Summa
        for(i=0;i<gpuSizeOperationWorld;i++)
        {
            int gpuRealId=MatrixUtilitiesCuda<Toperation>::getRealGpuId(i,gpuSizeSystem);
            rowColor=matrixA->calculateRowColor(i);
            columnColor=matrixA->calculateColumnColor(i);
            rowColorsLogic[rowColor].insert(i);
            columnColorsLogic[columnColor].insert(i);
            commElements[i]=new CommSummaElement(i,gpuRealId,rowColor,columnColor);
        }
        //Creación de los comunicadores
        std::set<int> rowsColorSet,columnColorSet;
        for(i=0;i<meshRowSize||i<meshColumnSize;i++)
        {
            if(i<meshRowSize)
            {
                rowsColorSet = rowColorsLogic[i];
                createNcclCommunicator(commElements,rowsColorSet,true);
            }
            if(i<meshColumnSize)
            {
                columnColorSet = columnColorsLogic[i];
                createNcclCommunicator(commElements,columnColorSet,false);
            }
        }
        summaComms[meshRowSize]=std::make_tuple(commElements,rowColorsLogic,columnColorsLogic);
    }
    return summaComms[meshRowSize];
}


template <class Toperation>
MatrixMain<Toperation>& NcclMultiplicationEnvironment<Toperation>::performCalculations(MatrixMain<Toperation>& ma,MatrixMain<Toperation>& mb)
{
    OperationProperties op;
    MatrixMain<Toperation> *mc;

    if(!MatrixUtilitiesCuda<Toperation>::canMultiply(ma.getColumnsReal(),mb.getRowsReal()))
    {
        throw std::invalid_argument("La operación no se puede realizar porque las columnas no coinciden con las filas. Columnas: " +std::to_string(ma.getColumnsReal())+ ", Filas: "+ std::to_string(mb.getRowsReal()));
    }

    //METER AQUÍ COMPROBACIÖN DE TAMAÑO Y SI ES MENOR HACERLA SECUENCIAL EN CASO DE QUE SE QUIERA HACER

    //Realización de la distribución pertinente
    if(ma.getIsDistributed()&&mb.getIsDistributed()&&ma.getBlockColumnSize()==mb.getBlockRowSize())
    {
        op.meshRowSize=ma.getMeshRowSize();
        op.meshColumnSize=ma.getMeshColumnSize();
    }else if(ma.getIsDistributed() && !mb.getIsDistributed())
    {
        op=MatrixUtilitiesCuda<Toperation>::getMeshAndMatrixSizeFromOneDistributedMatrix(ma.getRowsUsed(),ma.getColumnsUsed(), mb.getRowsReal(),mb.getColumnsReal(),ma.getMeshRowSize(),ma.getMeshColumnSize(),true);
        mb.setRowsUsed(op.columnsAorRowsB);
        mb.setColumnsUsed(op.columnsB);
        mb.setMatrixOperationProperties(op.meshRowSize,op.meshColumnSize,op.blockRowSizeB,op.blockColumnSizeB);
        mb.distributeMatrixIntoGpus();
        mb.waitAllStreamsOfAllWorkers();
    }else if(!ma.getIsDistributed() && mb.getIsDistributed())
    {
        op=MatrixUtilitiesCuda<Toperation>::getMeshAndMatrixSizeFromOneDistributedMatrix(ma.getRowsReal(),ma.getColumnsReal(), mb.getRowsUsed(),mb.getColumnsUsed(),mb.getMeshRowSize(),mb.getMeshColumnSize(),false);
        ma.setRowsUsed(op.rowsA);
        ma.setColumnsUsed(op.columnsAorRowsB);
        ma.setMatrixOperationProperties(op.meshRowSize,op.meshColumnSize,op.blockRowSizeA,op.blockColumnSizeA);
        ma.distributeMatrixIntoGpus();
        ma.waitAllStreamsOfAllWorkers();
    }else if(ma.getIsDistributed()&& mb.getIsDistributed()&&ma.getBlockColumnSize()!=mb.getBlockRowSize())
    {//Se decide recuperar b y redistribuirla. Puede que haya una mejor estrategia
        mb.getHostMatrix();
        mb.setIsDistributed(false);
        op=MatrixUtilitiesCuda<Toperation>::getMeshAndMatrixSizeFromOneDistributedMatrix(ma.getRowsUsed(),ma.getColumnsUsed(), mb.getRowsReal(),mb.getColumnsReal(),ma.getMeshRowSize(), ma.getMeshColumnSize(),true);
        mb.setRowsUsed(op.columnsAorRowsB);
        mb.setColumnsUsed(op.columnsB);
        mb.setMatrixOperationProperties(op.meshRowSize,op.meshColumnSize,op.blockRowSizeB,op.blockColumnSizeB);
        mb.distributeMatrixIntoGpus();
        mb.waitAllStreamsOfAllWorkers();
    }else if(!ma.getIsDistributed() && !mb.getIsDistributed())
    {
        op = MatrixUtilitiesCuda<Toperation>::getMeshAndMatrixSize(ma.getRowsReal(), ma.getColumnsReal(), mb.getRowsReal(), mb.getColumnsReal(), gpuSizeWorld);
        
        ma.setRowsUsed(op.rowsA);
        ma.setColumnsUsed(op.columnsAorRowsB);
        mb.setRowsUsed(op.columnsAorRowsB);
        mb.setColumnsUsed(op.columnsB);

        if (printMatrix)
        {
            std::cout << "NGpus: " << op.gpuSize << ", meshRowSize: " << op.meshRowSize << ", meshColumnSize: " << op.meshColumnSize << ", blockRowSizeA: " << \
            op.blockRowSizeA << ", blockColumnSizeA: " << op.blockColumnSizeA << ", blockRowSizeB: " << op.blockRowSizeB << ", blockColumnSizeB: " << \
            op.blockColumnSizeB << ", rowsA: " << op.rowsA << ", columnsAorRowsB: " << op.columnsAorRowsB << ", columnsB: " << op.columnsB << std::endl;

            std::cout << "A-> Rows: " << ma.getRowsReal() << ", Columns: " << ma.getColumnsReal() << ", Matriz A:" << std::endl;
            MatrixUtilitiesCuda<Toperation>::printMatrix(ma.getRowsReal(), ma.getColumnsReal(), ma.getHostMatrix());
            std::cout << "B-> Rows: " << mb.getRowsReal() << ", Columns: " << mb.getColumnsReal() << ", Matriz B:" << std::endl;
            MatrixUtilitiesCuda<Toperation>::printMatrix(mb.getRowsReal(), mb.getColumnsReal(), mb.getHostMatrix());
        }
        this->gpuSizeOperationWorld=op.gpuSize;
        this->gpuSizeOperationSystem=min(this->gpuSizeSystem,op.gpuSize);
        ma.setMatrixOperationProperties(op.meshRowSize,op.meshColumnSize,op.blockRowSizeA,op.blockColumnSizeA);
        mb.setMatrixOperationProperties(op.meshRowSize,op.meshColumnSize,op.blockRowSizeB,op.blockColumnSizeB);
        ma.distributeMatrixIntoGpus();
        mb.distributeMatrixIntoGpus();
        ma.waitAllStreamsOfAllWorkers();
        mb.waitAllStreamsOfAllWorkers();
    }else
    {
        throw std::runtime_error("Error. Se ha producido algún error en la librería");
    }

    mc=ncclSumma(&ma,&mb,op.meshRowSize,op.meshColumnSize);
    return *mc;
}

template <class Toperation>
MatrixMain<Toperation>*  NcclMultiplicationEnvironment<Toperation>::ncclSumma(MatrixMain<Toperation>* matrixA, MatrixMain<Toperation>* matrixB, int meshRowsSize, int meshColumnsSize)
{
    int i,vecI,gpuRank,rootRank;
    std::vector<int> vecOfActualComm;
    ncclComm_t commActual;cudaStream_t *streamComm;
    int rowsA = matrixA->getRowsUsed();
    int columnsAorRowsB = matrixA->getColumnsUsed();
    int columnsB = matrixB->getColumnsUsed();
    int blockSizeA = matrixA->getBlockSize();
    int blockSizeB = matrixB->getBlockSize();
    int blockRowSizeA = matrixA->getBlockRowSize();
    int blockColumnsSizeA = matrixA->getBlockColumnSize();
    int blockColumnsSizeB = matrixB->getBlockColumnSize();
    int blockRowSizeB = matrixB->getBlockRowSize();
    Toperation alphaGemm= matrixA->getAlphaGemm();
    //Creación del esquelo del elemento que va a ser devuelto
    MatrixMain<Toperation> *mc= new MatrixMain<Toperation>(this,matrixA->getRowsReal(),matrixB->getColumnsReal());
    mc->setRowsUsed(matrixA->getRowsUsed());
    mc->setColumnsUsed(matrixB->getColumnsUsed());
    mc->setMatrixOperationProperties(meshRowsSize,meshColumnsSize,blockRowSizeA,blockColumnsSizeB);
    mc->setIsDistributed(true);
    //Crear o recuperar los comunicadores
    auto commData=getOrCreateCommunicators(meshRowsSize,meshColumnsSize,matrixA);
    std::vector<CommSummaElement*> commElements=std::get<0>(commData);
    std::vector<std::set<int>> rowColorsLogic=std::get<1>(commData);
    std::vector<std::set<int>> columnColorsLogic=std::get<2>(commData);

    if(lastMeshRowSize!=meshRowsSize && lastMeshColumnSize!= meshColumnsSize && lastBlockRowSizeA!=blockRowSizeA 
        && lastBlockColumnSizeA!=blockColumnsSizeA  && lastBlockRowSizeB!=blockRowSizeB && lastBlockColumnSizeB!=blockColumnsSizeB )
    {
        eraseBufferMatrix();
        //Reserva de las matrices buffer para cada gpu
        for(i=0;i<gpuSizeOperationWorld;i++)
        {
            int gpuRealId=MatrixUtilitiesCuda<Toperation>::getRealGpuId(i,gpuSizeSystem);
            CUDACHECK(cudaSetDevice(gpuRealId));
            Toperation *gpuAuxA=MatrixUtilitiesCuda<Toperation>::cudaMatrixMemoryAllocationGPU(blockRowSizeA,blockColumnsSizeA,cublasStreams[gpuRealId]);
            Toperation *gpuAuxB=MatrixUtilitiesCuda<Toperation>::cudaMatrixMemoryAllocationGPU(blockRowSizeB,blockColumnsSizeB,cublasStreams[gpuRealId]);
            gpuAuxiliarMatricesA.push_back(gpuAuxA);gpuAuxiliarMatricesB.push_back(gpuAuxB);
        }
        lastMeshRowSize=meshRowsSize; lastMeshColumnSize=meshColumnsSize;
        lastBlockRowSizeA=blockRowSizeA; lastBlockColumnSizeA=blockColumnsSizeA;
        lastBlockRowSizeB=blockRowSizeB; lastBlockColumnSizeB=blockColumnsSizeB;
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
                CUDACHECK(cudaMemcpyAsync(gpuAuxiliarMatricesB[gpuRank],matrixB->getGpuWorkers()[gpuRank]->getMatrixLocal(i / meshRowsSize),blockSizeB*sizeof(Toperation),cudaMemcpyDeviceToDevice,*matrixB->getGpuWorkers()[gpuRank]->getStream(i / meshRowsSize)));
            }
        }
        //Esperar esa copia
        matrixA->waitAllStreamsOfAllWorkers();
        matrixB->waitAllStreamsOfAllWorkers();
        
        //Realizacion de las comunicaciones
        NCCLCHECK(ncclGroupStart());
        for(gpuRank=0;gpuRank<gpuSizeOperationWorld;gpuRank++)
	    {
            if(commElements[gpuRank]->getRankCommRowLogic()==(i % meshColumnsSize))
            {
                for(vecI=0;vecI<commElements[gpuRank]->getRowDevices().size();vecI++)
                {
                    vecOfActualComm=commElements[gpuRank]->getRowDevices()[vecI];
                    if(std::find(vecOfActualComm.begin(), vecOfActualComm.end(),gpuRank ) == vecOfActualComm.end())
                    {//Para gpus lógicas que no están físicas
                        vecOfActualComm.push_back(gpuRank);
                    }
                    for(int gpuIdComm:vecOfActualComm)
                    {
                        int realId=MatrixUtilitiesCuda<Toperation>::getRealGpuId(gpuIdComm,gpuSizeSystem);
                        CUDACHECK(cudaSetDevice(realId));
                        streamComm=commElements[gpuIdComm]->getStreamRow();
                        commActual=commElements[gpuIdComm]->getCommRow();
                        rootRank=commElements[gpuRank]->getRankCommRowPhysical();
                        if(MatrixUtilitiesCuda<Toperation>::getRealGpuId(vecOfActualComm[0],gpuSizeSystem)!=vecOfActualComm[0]&& vecI>0)
                        {//Para gpus lógicas que no están físicas
                            streamComm=commElements[gpuIdComm]->getStreamRowMySelf();
                            commActual=commElements[gpuIdComm]->getCommRowMySelf();
                            rootRank=0;
                        }
                        NCCLCHECK(ncclBroadcast(gpuAuxiliarMatricesA[gpuRank],gpuAuxiliarMatricesA[gpuIdComm],blockSizeA,
                            basicOperationType,rootRank,commActual,
                            *streamComm));
                    }
                }
            }
            if(commElements[gpuRank]->getRankCommColumnLogic()==(i % meshRowsSize))
            {
                for(vecI=0;vecI<commElements[gpuRank]->getColumnDevices().size();vecI++)
                {
                    vecOfActualComm=commElements[gpuRank]->getColumnDevices()[vecI];
                    if(std::find(vecOfActualComm.begin(), vecOfActualComm.end(),gpuRank ) == vecOfActualComm.end())
                    {//Para gpus lógicas que no están físicas
                        vecOfActualComm.push_back(gpuRank);
                    }
                    for(int gpuIdComm:vecOfActualComm)
                    {
                        int realId=MatrixUtilitiesCuda<Toperation>::getRealGpuId(gpuIdComm,gpuSizeSystem);
                        CUDACHECK(cudaSetDevice(realId));
                        streamComm=commElements[gpuIdComm]->getStreamColumn();
                        commActual=commElements[gpuIdComm]->getCommColumn();
                        rootRank=commElements[gpuRank]->getRankCommColumnPhysical();
                        if(MatrixUtilitiesCuda<Toperation>::getRealGpuId(vecOfActualComm[0],gpuSizeSystem)!=vecOfActualComm[0]&& vecI>0)
                        {//Para gpus lógicas que no están físicas. Importante el orden de estas dos instrucciones para que vaya bien el índice
                            streamComm=commElements[gpuIdComm]->getStreamColumnMySelf();
                            commActual=commElements[gpuIdComm]->getCommColumnMySelf();
                            rootRank=0;
                        }
                        NCCLCHECK(ncclBroadcast(gpuAuxiliarMatricesB[gpuRank],gpuAuxiliarMatricesB[gpuIdComm],blockSizeB,
                            basicOperationType,rootRank,commActual,
                            *streamComm));
                    }
                }
            }
        }
        NCCLCHECK(ncclGroupEnd());
        //Esperar las comunicaciones
        for(gpuRank=0;gpuRank<commElements.size();gpuRank++)
        {
            CUDACHECK(cudaSetDevice(MatrixUtilitiesCuda<Toperation>::getRealGpuId(gpuRank,gpuSizeSystem)));
            commElements[gpuRank]->waitStreams();
        }
        //Realización de todas las multiplicaciones
        for(gpuRank=0;gpuRank<gpuSizeOperationWorld;gpuRank++)
	    {
            int gpuRealId=MatrixUtilitiesCuda<Toperation>::getRealGpuId(gpuRank,gpuSizeSystem);
            CUDACHECK(cudaSetDevice(gpuRealId));
            MatrixUtilitiesCuda<Toperation>::matrixCublasMultiplication(cublasHandlers[gpuRealId],opType,blockRowSizeA,blockRowSizeB,blockColumnsSizeB,gpuAuxiliarMatricesA[gpuRank],gpuAuxiliarMatricesB[gpuRank],mc->getGpuWorkers()[gpuRank]->getMatrixLocal(0),alphaGemm,1.0);
        }
        waitAllCublasStreams();
    }
    return mc;
}

template class NcclMultiplicationEnvironment<double>;
template class NcclMultiplicationEnvironment<float>;

