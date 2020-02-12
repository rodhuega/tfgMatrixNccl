#include "MpiMatrix.h"
#include "MatrixUtilities.h"
#include <unistd.h>

MpiMatrix::MpiMatrix(int cpuSize,int cpuRank, int NSize)
{
    this->cpuRank = cpuRank;
    this->cpuSize=cpuSize;
    N = NSize;
    blockNSize = N / 2;
    blockSize = blockNSize * blockNSize;
    sendCounts.reserve(cpuSize);
    std::fill_n(sendCounts.begin(),cpuSize,1);
    //WIP: MAS PROCESOS
    blocks.push_back(0);
    blocks.push_back(blockNSize);
    blocks.push_back(N * blockNSize);
    blocks.push_back( N * blockNSize + blockNSize);
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

double *MpiMatrix::mpiDistributeMatrix(double *matrixGlobal, int root)
{
    double *globalptr = NULL;
    if (cpuRank == root)
    {
        globalptr = matrixGlobal;
    }
    //WIP: MAS PROCESOS
    int matrixLocalIndices[4] = {blocks[0], blocks[1], blocks[2], blocks[3]};
    double *matrixLocal = (double *)calloc(blockNSize * blockNSize, sizeof(double));
    MPI_Scatterv(globalptr, &sendCounts[0], matrixLocalIndices, matrixLocalType, matrixLocal, blockSize, MPI_DOUBLE, root, MPI_COMM_WORLD);
    return matrixLocal;
}

double *MpiMatrix::mpiRecoverDistributedMatrixGatherV(double *matrixLocal, int root)
{
    double *matrix = NULL;
    if (cpuRank == root)
    {
        matrix = (double *)calloc(N * N, sizeof(double));
    }
    MPI_Gatherv(matrixLocal, blockSize, MPI_DOUBLE, matrix, &sendCounts[0], &blocks[0], matrixLocalType, root, MPI_COMM_WORLD);
    // if(cpuRank==0) WIP: DESTRUCTOR
    // {
    //     MPI_Type_free(&matrixLocalType);
    // }
    return matrix;
}

double *MpiMatrix::mpiRecoverDistributedMatrixReduce(double *matrixLocal, int root)
{
    double *matrix = NULL;
    if (cpuRank == root)
    {
        matrix = (double *)calloc(N * N, sizeof(double));
    }
    int i;
    double *matrixLocalTotalNSize = (double *)calloc(N * N, sizeof(double));
    int initialBlockPosition = blocks[cpuRank];
    for (i = 0; i < blockNSize; i++)
    {
        memcpy(&matrixLocalTotalNSize[initialBlockPosition + i * N], &matrixLocal[i * blockNSize], blockNSize * sizeof(double));
    }
    MPI_Reduce(matrixLocalTotalNSize,matrix,N*N,MPI_DOUBLE,MPI_SUM,root,MPI_COMM_WORLD);
    return matrix;
}

double *MpiMatrix::mpiRecoverDistributedMatrixSendRec(double *matrixLocal, int root)
{

    return NULL;
}