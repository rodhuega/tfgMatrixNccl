#ifndef MpiMatrix_H
#define MpiMatrix_H

#include "mpi.h"
#include "vector"
#include "MatrixUtilities.h"
#include <unistd.h>
#include <cblas.h>
class MpiMatrix
{
private:
    MPI_Datatype matrixLocalType;
    double* matrixLocal;
    int rowSize,columnSize,blockRowSize,blockColumnSize,blockSize,cpuRank,cpuSize,meshRowSize,meshColumnSize;
    std::vector<int> sendCounts;
    std::vector<int> blocks;

public:
    MpiMatrix(int cpuSize,int cpuRank,int meshRowSize,int meshColumnSize,int rowSize,int columnSize);
    int getBlockRowSize();
    int getBlockColumnSize();
    int getRowSize();
    int getColumnSize();
    int getMeshRowSize();
    int getMeshColumnSize();
    int getBlockSize();
    void setMatrixLocal(double* matrixLocal);
    double* getMatrixLocal();
    void mpiDistributeMatrix(double *matrixGlobal,int root);
    double *mpiRecoverDistributedMatrixGatherV(int root);
    double* mpiRecoverDistributedMatrixReduce(int root);
};
#endif