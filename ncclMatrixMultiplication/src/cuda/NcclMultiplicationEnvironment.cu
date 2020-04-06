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
        ma->setBlockAndMeshSize(op.meshRowSize,op.meshColumnSize,op.blockRowSizeA,op.blockColumnSizeA);
        mb->setBlockAndMeshSize(op.meshRowSize,op.meshColumnSize,op.blockRowSizeB,op.blockColumnSizeB);
        ma->distributeMatrixIntoGpus();
        mb->distributeMatrixIntoGpus();
        ma->waitAllStreamsOfAllWorkers();
        mb->waitAllStreamsOfAllWorkers();
        MatrixUtilitiesCuda<Toperation>::cudaDebugMatricesLocalDifferentGpuWorkers(gpuSizeOperationWorld,op.blockRowSizeA,op.blockColumnSizeA,ma->getGpuWorkers());

        Toperation *matrixLocalC;
        //W.I.P CREO QUE ESTE NEW IRA DENTRO DEL SUMA
        mc=new MatrixMain<Toperation>(this,"C",ma->getRowsReal(), mb->getColumnsReal());
        mc->setRowsUsed(ma->getRowsUsed());
        mc->setColumnsUsed(mb->getColumnsUsed());
        std::cout<<"Llegamos sin problemas"<<std::endl;
    }


}


template class NcclMultiplicationEnvironment<double>;
