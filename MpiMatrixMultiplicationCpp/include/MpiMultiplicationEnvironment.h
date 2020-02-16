#ifndef MpiMultiplicationEnvironment_H
#define MpiMultiplicationEnvironment_H

#include "mpi.h"
#include "vector"
#include "MatrixUtilities.h"
#include <unistd.h>
#include <cblas.h>
class MpiMultiplicationEnvironment
{
private:
    MPI_Datatype matrixLocalType;
    int rowSize,columnSize,blockRowSize,blockColumnSize,blockSize,cpuRank,cpuSize,meshRowSize,meshColumnSize;
    std::vector<int> sendCounts;
    std::vector<int> blocks;

public:
    MpiMultiplicationEnvironment(int cpuSize,int cpuRank,int meshRowSize,int meshColumnSize,int rowSize,int columnSize);
    int getBlockRowSize();
    int getBlockColumnSize();
    double *mpiDistributeMatrix(double *matrixGlobal,int root);
    double *mpiRecoverDistributedMatrixGatherV(double *matrixLocal,int root);
    double* mpiRecoverDistributedMatrixReduce(double* matrixLocal,int root);
    double* mpiSumma(int rowsA,int columnsAorRowsB,int columnsB,double* matrixLocalA, double *matrixLocalB,int meshRowsSize, int meshColumnsSize);
};
#endif