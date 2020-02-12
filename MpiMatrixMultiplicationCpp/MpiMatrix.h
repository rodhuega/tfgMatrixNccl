#ifndef MpiMatrix_H
#define MpiMatrix_H
#include "mpi.h"
#include "vector"
class MpiMatrix
{
private:
    MPI_Datatype matrixLocalType;
    int N,blockNSize,blockSize,cpuRank,cpuSize;
    std::vector<int> sendCounts;
    std::vector<int> blocks;

public:
    MpiMatrix(int cpuSize,int cpuRank,int NSize);
    double *mpiDistributeMatrix(double *matrixGlobal,int root);
    double *mpiRecoverDistributedMatrixGatherV(double *matrixLocal,int root);
    double* mpiRecoverDistributedMatrixReduce(double* matrixLocal,int root);
    double* mpiRecoverDistributedMatrixSendRec(double* matrixLocal,int root);
};
#endif