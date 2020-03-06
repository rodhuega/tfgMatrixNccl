#include "MpiMatrix.h"

template <class Toperation>
MpiMatrix<Toperation>::MpiMatrix(int cpuSize,int cpuRank,int meshRowSize,int meshColumnSize,MatrixMain<Toperation>* matrixGlobal,MPI_Comm commOperation,MPI_Datatype basicOperationType)
{
    this->cpuRank = cpuRank;
    this->cpuSize = cpuSize;
    this->meshRowSize = meshRowSize;
    this->meshColumnSize = meshColumnSize;
    this->commOperation = commOperation;
    this->basicOperationType = basicOperationType;
    this->matrixMainGlobal=matrixGlobal;
    //calculo de los tamaños de la matriz de forma local
    blockRowSize = matrixMainGlobal->getRowsUsed() / meshRowSize;
    blockColumnSize = matrixMainGlobal->getColumnsUsed() / meshColumnSize;
    blockSize = blockRowSize * blockColumnSize;
    //Creamos un tipo especifico de mpi para distribuir la matriz o recuperarla con gatherV
    sendCounts.reserve(cpuSize);
    std::fill_n(sendCounts.begin(), cpuSize, 1);
    int i, posColumnBelong, posRowBelong;
    //Asignamos en que posicion empieza cada bloque
    for (i = 0; i < cpuSize; i++)
    {
        posRowBelong = (i / meshColumnSize) * matrixMainGlobal->getColumnsUsed() * blockRowSize;
        posColumnBelong = (i % meshColumnSize) * blockColumnSize;
        blocks.push_back(posColumnBelong + posRowBelong);
    }
    
    if (cpuRank == 0)
    {
        int sizes[2] = {matrixMainGlobal->getRowsUsed(), matrixMainGlobal->getColumnsUsed()};
        int subsizes[2] = {blockRowSize, blockColumnSize};
        int starts[2] = {0, 0};
        MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, basicOperationType, &matrixLocalType);
        int typeSize;
        MPI_Type_size(basicOperationType, &typeSize);
        MPI_Type_create_resized(matrixLocalType, 0, 1 * typeSize, &matrixLocalType);
        MPI_Type_commit(&matrixLocalType);
    }
    columnColor= calculateColumnColor(cpuRank);
    rowColor=calculateRowColor(cpuRank);
}

template <class Toperation>
MpiMatrix<Toperation>::~MpiMatrix()
{
    //WIP: DESTRUCTOR, no se porque falla en esta linea al intentar liberar la memoria
    // MatrixUtilities<Toperation>::matrixFree(matrixLocal);
    if (cpuRank == 0)
    {
        MPI_Type_free(&matrixLocalType);
    }
}

template <class Toperation>
int MpiMatrix<Toperation>::getBlockRowSize()
{
    return blockRowSize;
}
template <class Toperation>
int MpiMatrix<Toperation>::getBlockColumnSize()
{
    return blockColumnSize;
}
template <class Toperation>
int MpiMatrix<Toperation>::getMeshRowSize()
{
    return meshRowSize;
}
template <class Toperation>
int MpiMatrix<Toperation>::getMeshColumnSize()
{
    return meshColumnSize;
}
template <class Toperation>
int MpiMatrix<Toperation>::getBlockSize()
{
    return blockSize;
}
template <class Toperation>
Toperation *MpiMatrix<Toperation>::getMatrixLocal()
{
    return matrixLocal;
}
template <class Toperation>
void MpiMatrix<Toperation>::setMatrixLocal(Toperation *matrixLocal)
{
    this->matrixLocal = matrixLocal;
}

template <class Toperation>
MatrixMain<Toperation>* MpiMatrix<Toperation>::getMatrixMain()
{
    return matrixMainGlobal;
}

template <class Toperation>
void MpiMatrix<Toperation>::mpiDistributeMatrixSendRecv(Toperation *matrixGlobal, int root)
{
    int i,blockColumnSizeSend,blockRowSizeSend;
    Toperation *globalptr = NULL;
    if (cpuRank == root)
    {
        globalptr = matrixGlobal;
    }
    matrixLocal = MatrixUtilities<Toperation>::matrixMemoryAllocation(blockRowSize, blockColumnSize);
    if(cpuRank==root)
    {
        //Faltaria comprobar si es caso de columna extendida para que envie menos datos.
        blockColumnSizeSend=blockColumnSizeSend=(columnColor==(meshColumnSize-1)) ? blockColumnSize : (blockColumnSize-(matrixMainGlobal->getColumnsUsed()-matrixMainGlobal->getColumnsReal()));
        for(i=0;i<blockRowSize;i++)
        {
            memcpy(&matrixLocal[i*blockColumnSize],&matrixGlobal[blocks[0]+i*matrixMainGlobal->getColumnsUsed()],sizeof(Toperation)*blockColumnSizeSend);
        }
        int j;
        for(j=1;j<cpuSize;j++)
        {
            blockColumnSizeSend=(calculateColumnColor(j)!=(meshColumnSize-1)) ? blockColumnSize : (blockColumnSize-(matrixMainGlobal->getColumnsUsed()-matrixMainGlobal->getColumnsReal()));
            blockRowSizeSend=(calculateRowColor(j)!=(meshRowSize-1)) ? blockRowSize : (blockRowSize-(matrixMainGlobal->getRowsUsed()-matrixMainGlobal->getRowsReal()));
            for(i=0;i<blockRowSizeSend;i++)
            {
                //Faltaria comprobar si es caso de columna extendida para que envie menos datos.
                MPI_Send(&matrixGlobal[blocks[j]+i*matrixMainGlobal->getColumnsUsed()],blockColumnSizeSend,basicOperationType,j,root,commOperation);
            }
        }
    }else
    {
        blockColumnSizeSend=(columnColor!=(meshColumnSize-1)) ? blockColumnSize : (blockColumnSize-(matrixMainGlobal->getColumnsUsed()-matrixMainGlobal->getColumnsReal()));
        blockRowSizeSend=(rowColor!=(meshRowSize-1)) ? blockRowSize : (blockRowSize-(matrixMainGlobal->getRowsUsed()-matrixMainGlobal->getRowsReal()));
        for(i=0;i<blockRowSizeSend;i++)
        {            
            MPI_Recv(&matrixLocal[i*blockColumnSize],blockColumnSizeSend,basicOperationType,root,root,commOperation,MPI_STATUS_IGNORE);
        }
    }
}

template <class Toperation>
void MpiMatrix<Toperation>::mpiDistributeMatrix(Toperation *matrixGlobal, int root)
{
    Toperation *globalptr = NULL;
    if (cpuRank == root)
    {
        globalptr = matrixGlobal;
    }
    matrixLocal = MatrixUtilities<Toperation>::matrixMemoryAllocation(blockRowSize, blockColumnSize);
    MPI_Scatterv(globalptr, &sendCounts[0], &blocks[0], matrixLocalType, matrixLocal, blockSize, basicOperationType, root, commOperation);
}
template <class Toperation>
Toperation *MpiMatrix<Toperation>::mpiRecoverDistributedMatrixGatherV(int root)
{
    Toperation *matrix = NULL;
    if (cpuRank == root)
    {
        matrix = MatrixUtilities<Toperation>::matrixMemoryAllocation(matrixMainGlobal->getRowsUsed(), matrixMainGlobal->getColumnsUsed());
    }
    
    MPI_Gatherv(matrixLocal, blockSize, basicOperationType, matrix, &sendCounts[0], &blocks[0], matrixLocalType, root, commOperation);
    return matrix;
}
template <class Toperation>
Toperation *MpiMatrix<Toperation>::mpiRecoverDistributedMatrixReduce(int root)
{
    Toperation *matrix = NULL;
    int i;
    Toperation *matrixLocalTotalNSize = MatrixUtilities<Toperation>::matrixMemoryAllocation(matrixMainGlobal->getRowsUsed(), matrixMainGlobal->getColumnsUsed());
    int initialBlockPosition = blocks[cpuRank];
    if (cpuRank == root)
    {
        matrix = MatrixUtilities<Toperation>::matrixMemoryAllocation(matrixMainGlobal->getRowsUsed(), matrixMainGlobal->getColumnsUsed());
    }
    for (i = 0; i < blockRowSize; i++)
    {
        memcpy(&matrixLocalTotalNSize[initialBlockPosition + i * matrixMainGlobal->getColumnsUsed()], &matrixLocal[i * blockColumnSize], blockColumnSize * sizeof(Toperation));
    }
    MPI_Reduce(matrixLocalTotalNSize, matrix, matrixMainGlobal->getRowsUsed() * matrixMainGlobal->getColumnsUsed(), basicOperationType, MPI_SUM, root, commOperation);
    return matrix;
}

template <class Toperation>
int MpiMatrix<Toperation>::calculateRowColor(int cpuRank)
{
    return cpuRank / meshColumnSize;
}

template <class Toperation>
int MpiMatrix<Toperation>::calculateColumnColor(int cpuRank)
{
    return cpuRank % meshColumnSize;
}

template <class Toperation>
int MpiMatrix<Toperation>::getRowColor()
{
    return rowColor;
}

template <class Toperation>
int MpiMatrix<Toperation>::getColumnColor()
{
    return columnColor;
}

template class MpiMatrix<double>;
template class MpiMatrix<float>;