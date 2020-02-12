#ifndef MpiMatrix_H
#define MpiMatrix_H
#include "mpi.h"
class MpiMatrix
{
private:
    MPI_Datatype matrixLocalType;
    int N,blockNSize,blockSize,cpuRank;
    int* sendCounts;

public:
    MpiMatrix(int cpuRank,int NSize);
    double *mpiDistributeMatrix(double *matrixGlobal);
    double *mpiRecoverDistributedMatrixGatherV(double *localMatrix);
};
#endif