#include "MpiMatrix.h"
#include "MatrixUtilities.h"

MpiMatrix::MpiMatrix(int cpuRank,int NSize)
{
    this->cpuRank=cpuRank;
    N=NSize;
    blockNSize = N / 2;
    blockSize = blockNSize * blockNSize;
    if (cpuRank == 0)
    {
        int sizes[2] = {N, N};
        int subsizes[2] = {blockNSize, blockNSize};
        int starts[2] = {0, 0};
        MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &matrixLocalType);
        int doubleSize;
        MPI_Type_size(MPI_DOUBLE, &doubleSize);
        MPI_Type_create_resized(matrixLocalType, 0, 1 * doubleSize, &matrixLocalType);
        MPI_Type_commit(&matrixLocalType);
    }
}

double *MpiMatrix::mpiDistributeMatrix(double *matrixGlobal)
{
    double *globalptr = NULL;
    if (cpuRank == 0)
    {
        globalptr = matrixGlobal;
    }
    const int blocks[4] = {0, blockNSize, N * blockNSize, N * blockNSize + blockNSize};
    int sendCounts[4] = {1, 1, 1, 1};
    int matrixLocalIndices[4] = {blocks[0], blocks[1], blocks[2], blocks[3]};
    double *matrixLocal = (double *)calloc(blockNSize * blockNSize, sizeof(double));
    MPI_Scatterv(globalptr, sendCounts, matrixLocalIndices, matrixLocalType, matrixLocal, blockSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return matrixLocal;
}

double* MpiMatrix::mpiRecoverDistributedMatrixGatherV(double* localMatrix)
{
    double *matrix=(double*)calloc(N*N,sizeof(double));
    const int blocks[4] = {0, blockNSize, N * blockNSize, N * blockNSize + blockNSize};

    int sendCounts[4] = {1, 1, 1, 1};
    MPI_Gatherv(localMatrix, blockSize, MPI_DOUBLE, matrix, sendCounts, blocks, matrixLocalType, 0, MPI_COMM_WORLD);
    // if(cpuRank==0) WIP:DESTRUCTOR
    // {
    //     MPI_Type_free(&matrixLocalType);
    // }
    return matrix;
}