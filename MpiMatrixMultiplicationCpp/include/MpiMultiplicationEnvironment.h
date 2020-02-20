#ifndef MpiMultiplicationEnvironment_H
#define MpiMultiplicationEnvironment_H

#include "MatrixUtilities.h"
#include "MpiMatrix.h"

template <class Toperation>
class MpiMultiplicationEnvironment
{
private:
    MPI_Datatype basicOperationType;
    MPI_Comm commOperation;
    int cpuRank,cpuSize;
    

public:
    MpiMultiplicationEnvironment(int cpuRank,int cpuSize,MPI_Comm commOperation,MPI_Datatype basicOperationType);
    MpiMatrix<Toperation> mpiSumma(MpiMatrix<Toperation> matrixLocalA, MpiMatrix<Toperation> matrixLocalB, int meshRowsSize, int meshColumnsSize);
};

#endif