#include "MpiMatrix.h"

MpiMatrix::MpiMatrix(int cpuSize, int cpuRank, int meshRowSize, int meshColumnSize, int rowSize, int columnSize)
{
    this->cpuRank = cpuRank;
    this->cpuSize = cpuSize;
    this->meshRowSize = meshRowSize;
    this->meshColumnSize = meshColumnSize;
    this->rowSize = rowSize;
    this->columnSize = columnSize;
    blockRowSize = rowSize / meshRowSize;
    blockColumnSize = columnSize / meshColumnSize;
    blockSize = blockRowSize * blockColumnSize;
    sendCounts.reserve(cpuSize);
    std::fill_n(sendCounts.begin(), cpuSize, 1);
    int i, posColumnBelong, posRowBelong;
    //Asignamos en que posicion empieza cada bloque
    for (i = 0; i < cpuSize; i++)
    {
        posRowBelong = (i / meshColumnSize) * columnSize * blockRowSize;
        posColumnBelong = (i % meshColumnSize) * blockColumnSize;
        blocks.push_back(posColumnBelong + posRowBelong);
    }
    if (cpuRank == 0)
    {
        int sizes[2] = {rowSize, columnSize};
        int subsizes[2] = {blockRowSize, blockColumnSize};
        int starts[2] = {0, 0};
        MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &matrixLocalType);
        int doubleSize;
        MPI_Type_size(MPI_DOUBLE, &doubleSize);
        MPI_Type_create_resized(matrixLocalType, 0, 1 * doubleSize, &matrixLocalType);
        MPI_Type_commit(&matrixLocalType);
    }
}

int MpiMatrix::getBlockRowSize()
{
    return blockRowSize;
}

int MpiMatrix::getBlockColumnSize()
{
    return blockColumnSize;
}

int MpiMatrix::getRowSize()
{
    return rowSize;
}
int MpiMatrix::getColumnSize()
{
    return columnSize;
}

int MpiMatrix::getMeshRowSize()
{
    return meshRowSize;
}
int MpiMatrix::getMeshColumnSize()
{
    return meshColumnSize;
}

int MpiMatrix::getBlockSize()
{
    return blockSize;
}

double* MpiMatrix::getMatrixLocal()
{
    return matrixLocal;
}
void MpiMatrix::setMatrixLocal(double* matrixLocal)
{
    this->matrixLocal=matrixLocal;
}

void MpiMatrix::mpiDistributeMatrix(double *matrixGlobal, int root)
{
    double *globalptr = NULL;
    if (cpuRank == root)
    {
        globalptr = matrixGlobal;
    }
    matrixLocal = MatrixUtilities::matrixMemoryAllocation(blockRowSize, blockColumnSize);
    MPI_Scatterv(globalptr, &sendCounts[0], &blocks[0], matrixLocalType, matrixLocal, blockSize, MPI_DOUBLE, root, MPI_COMM_WORLD);
}

double *MpiMatrix::mpiRecoverDistributedMatrixGatherV(int root)
{
    double *matrix = NULL;
    if (cpuRank == root)
    {
        matrix = MatrixUtilities::matrixMemoryAllocation(rowSize, columnSize);
    }
    MPI_Gatherv(matrixLocal, blockSize, MPI_DOUBLE, matrix, &sendCounts[0], &blocks[0], matrixLocalType, root, MPI_COMM_WORLD);
    // if(cpuRank==0) WIP: DESTRUCTOR
    // {
    //     MPI_Type_free(&matrixLocalType);
    // }
    return matrix;
}

double *MpiMatrix::mpiRecoverDistributedMatrixReduce(int root)
{
    double *matrix = NULL;
    int i;
    double *matrixLocalTotalNSize = MatrixUtilities::matrixMemoryAllocation(rowSize, columnSize);
    int initialBlockPosition = blocks[cpuRank];

    if (cpuRank == root)
    {
        matrix = MatrixUtilities::matrixMemoryAllocation(rowSize, columnSize);
    }

    for (i = 0; i < blockRowSize; i++)
    {
        memcpy(&matrixLocalTotalNSize[initialBlockPosition + i * columnSize], &matrixLocal[i * blockColumnSize], blockColumnSize * sizeof(double));
    }
    MPI_Reduce(matrixLocalTotalNSize, matrix, rowSize * columnSize, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
    return matrix;
}
