#include "MpiMatrix.h"

template <class Toperation>
MpiMatrix<Toperation>::MpiMatrix(int cpuSize, int cpuRank, int meshRowSize, int meshColumnSize, int blockRowSize, int blockColumnSize, MatrixMain<Toperation> *matrixGlobal, MPI_Comm commOperation, MPI_Datatype basicOperationType)
{
    this->cpuRank = cpuRank;
    this->cpuSize = cpuSize;
    this->meshRowSize = meshRowSize;
    this->meshColumnSize = meshColumnSize;
    this->commOperation = commOperation;
    this->basicOperationType = basicOperationType;
    this->matrixMainGlobal = matrixGlobal;
    //calculo de los tamaños de la matriz de forma local
    this->blockRowSize = blockRowSize;
    this->blockColumnSize = blockColumnSize;
    numberOfRowBlocks = ceil(matrixGlobal->getRowsUsed() / blockRowSize);
    numberOfColumnBlocks = ceil(matrixGlobal->getColumnsUsed() / blockColumnSize);
    numberOfTotalBlocks = numberOfRowBlocks * numberOfColumnBlocks;
    blockSize = blockRowSize * blockColumnSize;
    //Creamos un tipo especifico de mpi para distribuir la matriz o recuperarla con gatherV
    sendCounts.reserve(cpuSize);
    std::fill_n(sendCounts.begin(), cpuSize, 1);
    int i, posColumnBelong, posRowBelong;
    //Asignamos en que posicion empieza cada bloque
    for (i = 0; i < numberOfTotalBlocks; i++)
    {
        posRowBelong = (i / meshColumnSize) * matrixMainGlobal->getColumnsReal() * blockRowSize;
        posColumnBelong = (i % meshColumnSize) * blockColumnSize;
        blocks.push_back(posColumnBelong + posRowBelong);
    }
    //ESTO PUEDE SOBRAR POR DEPRECATED
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
    columnColor = calculateColumnColor(cpuRank);
    rowColor = calculateRowColor(cpuRank);
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
Toperation *MpiMatrix<Toperation>::getMatrixLocal(int pos)
{
    return matricesLocal[pos];
}
template <class Toperation>
std::vector<Toperation *> MpiMatrix<Toperation>::getMatricesLocal()
{
    return matricesLocal;
}

template <class Toperation>
void MpiMatrix<Toperation>::setMatrixLocal(Toperation *matrixLocal)
{
    matricesLocal.push_back(matrixLocal);
}

template <class Toperation>
MatrixMain<Toperation> *MpiMatrix<Toperation>::getMatrixMain()
{
    return matrixMainGlobal;
}

template <class Toperation>
int MpiMatrix<Toperation>::calculateBlockDimensionSizeSendRec(int color, int meshDimensionSize, int blockDimenensionSize, int dimensionUsed, int dimensionReal)
{
    return (color != (meshDimensionSize - 1)) ? blockDimenensionSize : (blockDimenensionSize - (dimensionUsed - dimensionReal));
}

template <class Toperation>
void MpiMatrix<Toperation>::mpiDistributeMatrixSendRecv(Toperation *matrixGlobal, int root)
{
    int i, j, blockColumnSizeSend, blockRowSizeSend;
    Toperation *globalptr = NULL;
    if (cpuRank == root)
    {
        globalptr = matrixGlobal;
    }
    //Reserva de memoria de las matrices locales
    for (i = cpuRank; i < numberOfTotalBlocks; i += cpuSize)
    {
        matricesLocal.push_back(MatrixUtilities<Toperation>::matrixMemoryAllocation(blockRowSize, blockColumnSize));
    }
    //Distribucion de la informacion
    Toperation* actualLocalMatrix;
    if (cpuRank == root)
    {
        blockColumnSizeSend = calculateBlockDimensionSizeSendRec(columnColor, numberOfColumnBlocks, blockColumnSize, matrixMainGlobal->getColumnsUsed(), matrixMainGlobal->getColumnsReal());
        for (i = 0; i < blockRowSize; i++)
        {
            for(j=0;(cpuRank+j*cpuSize)<numberOfTotalBlocks;j++)
            {
                actualLocalMatrix=matricesLocal[j];
                memcpy(&actualLocalMatrix[i * blockColumnSize], &matrixGlobal[blocks[cpuRank+j*cpuSize] + i * matrixMainGlobal->getColumnsReal()], sizeof(Toperation) * blockColumnSizeSend);
            }
        }
        int cpuToSend,indexBlock;
        for (cpuToSend = 1; cpuToSend < cpuSize; cpuToSend++)
        {
            blockColumnSizeSend = calculateBlockDimensionSizeSendRec(calculateColumnColor(cpuToSend), numberOfColumnBlocks, blockColumnSize, matrixMainGlobal->getColumnsUsed(), matrixMainGlobal->getColumnsReal());
            blockRowSizeSend = calculateBlockDimensionSizeSendRec(calculateRowColor(cpuToSend), numberOfRowBlocks, blockRowSize, matrixMainGlobal->getRowsUsed(), matrixMainGlobal->getRowsReal());
            for (indexBlock=cpuToSend;indexBlock<numberOfTotalBlocks;indexBlock+=cpuSize)
            {
                for(i = 0; i < blockRowSizeSend; i++)
                {
                    MPI_Send(&matrixGlobal[blocks[indexBlock] + i * matrixMainGlobal->getColumnsReal()], blockColumnSizeSend, basicOperationType, cpuToSend, root, commOperation);
                }
            }
        }
    }
    else
    {
        for (j=0;(cpuRank+j*cpuSize)<numberOfTotalBlocks;j++)
        {
            blockRowSizeSend = calculateBlockDimensionSizeSendRec(calculateRowColor((cpuRank+j*cpuSize)), numberOfRowBlocks, blockRowSize, matrixMainGlobal->getRowsUsed(), matrixMainGlobal->getRowsReal());
            blockColumnSizeSend = calculateBlockDimensionSizeSendRec(calculateColumnColor((cpuRank+j*cpuSize)), numberOfColumnBlocks, blockColumnSize, matrixMainGlobal->getColumnsUsed(), matrixMainGlobal->getColumnsReal());
            for (i = 0; i < blockRowSizeSend; i++)
            {
                actualLocalMatrix=matricesLocal[j];
                MPI_Recv(&actualLocalMatrix[i * blockColumnSize], blockColumnSizeSend, basicOperationType, root, root, commOperation, MPI_STATUS_IGNORE);
            }
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
    matricesLocal.push_back(MatrixUtilities<Toperation>::matrixMemoryAllocation(blockRowSize, blockColumnSize));
    MPI_Scatterv(globalptr, &sendCounts[0], &blocks[0], matrixLocalType, matricesLocal[0], blockSize, basicOperationType, root, commOperation);
}
template <class Toperation>
Toperation *MpiMatrix<Toperation>::mpiRecoverDistributedMatrixGatherV(int root)
{
    Toperation *matrix = NULL;
    if (cpuRank == root)
    {
        matrix = MatrixUtilities<Toperation>::matrixMemoryAllocation(matrixMainGlobal->getRowsUsed(), matrixMainGlobal->getColumnsUsed());
    }

    MPI_Gatherv(matricesLocal[0], blockSize, basicOperationType, matrix, &sendCounts[0], &blocks[0], matrixLocalType, root, commOperation);
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
        memcpy(&matrixLocalTotalNSize[initialBlockPosition + i * matrixMainGlobal->getColumnsUsed()], &matricesLocal[0][i * blockColumnSize], blockColumnSize * sizeof(Toperation));
    }
    MPI_Reduce(matrixLocalTotalNSize, matrix, matrixMainGlobal->getRowsUsed() * matrixMainGlobal->getColumnsUsed(), basicOperationType, MPI_SUM, root, commOperation);
    return matrix;
}

template <class Toperation>
Toperation *MpiMatrix<Toperation>::mpiRecoverDistributedMatrixSendRecv(int root)
{
    Toperation* actualLocalMatrix;
    Toperation *matrix = NULL;
    int i, blockColumnSizeSend, blockRowSizeSend;
    Toperation *matrixLocalTotalNSize = MatrixUtilities<Toperation>::matrixMemoryAllocation(matrixMainGlobal->getRowsUsed(), matrixMainGlobal->getColumnsUsed());
    int initialBlockPosition = blocks[cpuRank];
    //Reserva de espacio para la matriz total
    if (cpuRank == root)
    {
        matrix = MatrixUtilities<Toperation>::matrixMemoryAllocation(matrixMainGlobal->getRowsReal(), matrixMainGlobal->getColumnsReal());
    }
    //Distribucion
    if (cpuRank == root)
    {
        blockColumnSizeSend = calculateBlockDimensionSizeSendRec(columnColor, meshColumnSize, blockColumnSize, matrixMainGlobal->getColumnsUsed(), matrixMainGlobal->getColumnsReal());
        for (i = 0; i < blockRowSize; i++)
        {
            actualLocalMatrix=matricesLocal[0];
            memcpy(&matrix[blocks[0] + i * matrixMainGlobal->getColumnsReal()], &actualLocalMatrix[i * blockColumnSize], sizeof(Toperation) * blockColumnSizeSend);
        }
        int j;
        for (j = 1; j < cpuSize; j++)
        {
            blockColumnSizeSend = calculateBlockDimensionSizeSendRec(calculateColumnColor(j), meshColumnSize, blockColumnSize, matrixMainGlobal->getColumnsUsed(), matrixMainGlobal->getColumnsReal());
            blockRowSizeSend = calculateBlockDimensionSizeSendRec(calculateRowColor(j), meshRowSize, blockRowSize, matrixMainGlobal->getRowsUsed(), matrixMainGlobal->getRowsReal());
            for (i = 0; i < blockRowSizeSend; i++)
            {
                MPI_Recv(&matrix[blocks[j] + i * matrixMainGlobal->getColumnsReal()], blockColumnSizeSend, basicOperationType, j, root, commOperation, MPI_STATUS_IGNORE);
            }
        }
    }
    else
    {
        blockColumnSizeSend = calculateBlockDimensionSizeSendRec(columnColor, meshColumnSize, blockColumnSize, matrixMainGlobal->getColumnsUsed(), matrixMainGlobal->getColumnsReal());
        blockRowSizeSend = calculateBlockDimensionSizeSendRec(rowColor, meshRowSize, blockRowSize, matrixMainGlobal->getRowsUsed(), matrixMainGlobal->getRowsReal());
        for (i = 0; i < blockRowSizeSend; i++)
        {
            actualLocalMatrix=matricesLocal[0];
            MPI_Send(&actualLocalMatrix[i * blockColumnSize], blockColumnSizeSend, basicOperationType, root, root, commOperation);
        }
    }
    return matrix;
}

template <class Toperation>
int MpiMatrix<Toperation>::calculateRowColor(int cpuRank)
{
    return cpuRank / numberOfColumnBlocks;
}

template <class Toperation>
int MpiMatrix<Toperation>::calculateColumnColor(int cpuRank)
{
    return cpuRank % numberOfColumnBlocks;
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