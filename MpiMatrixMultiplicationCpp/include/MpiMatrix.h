#ifndef MpiMatrix_H
#define MpiMatrix_H

#include "mpi.h"
#include "vector"
#include "MatrixUtilities.h"
#include <unistd.h>
#include <cblas.h>

template <class Toperation>
class MpiMatrix
{
private:
    MPI_Comm commOperation;
    MPI_Datatype basicOperationType,matrixLocalType;
    Toperation* matrixLocal;
    int rowSize,columnSize,blockRowSize,blockColumnSize,blockSize,cpuRank,cpuSize,meshRowSize,meshColumnSize;
    std::vector<int> sendCounts;
    std::vector<int> blocks;

public:
    MpiMatrix(int cpuSize,int cpuRank,int meshRowSize,int meshColumnSize,int rowSize,int columnSize,MPI_Comm commOperation,MPI_Datatype basicOperationType);
    int getBlockRowSize();
    int getBlockColumnSize();
    int getRowSize();
    int getColumnSize();
    int getMeshRowSize();
    int getMeshColumnSize();
    int getBlockSize();
    void setMatrixLocal(Toperation* matrixLocal);
    Toperation* getMatrixLocal();
    void mpiDistributeMatrix(Toperation *matrixGlobal,int root);
    Toperation *mpiRecoverDistributedMatrixGatherV(int root);
    Toperation* mpiRecoverDistributedMatrixReduce(int root);
};
#endif