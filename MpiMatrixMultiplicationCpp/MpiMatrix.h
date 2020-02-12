#ifndef MpiMatrix_H
#define MpiMatrix_H
#include "mpi.h"
class MpiMatrix
{
private:
    MPI_Datatype matrixLocalType;
    int N,blockNSize,blockSize,cpuRank;

public:
    MpiMatrix(int cpuRank,int NSize);
    double *mpiDistributeMatrix(double *matrixGlobal);
    double *mpiRecoverDistributedMatrixGatherV(int N, int cpuRank, double *localMatrix);
};
#endif