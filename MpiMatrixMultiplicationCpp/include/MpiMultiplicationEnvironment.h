#ifndef MpiMultiplicationEnvironment_H
#define MpiMultiplicationEnvironment_H

#include "MatrixUtilities.h"
#include "MpiMatrix.h"

template <class Toperation>
class MpiMultiplicationEnvironment
{
private:
    MPI_Comm commOperation;
    int cpuRank,cpuSize;
    

public:
    MpiMultiplicationEnvironment(int cpuRank,int cpuSize,MPI_Comm commOperation);
    MpiMatrix<Toperation> mpiSumma(MpiMatrix<Toperation> matrixLocalA, MpiMatrix<Toperation> matrixLocalB, int meshRowsSize, int meshColumnsSize);
    void Multiplicacion(int rowsA,int columnsAorRowsB,int columnsB,Toperation* A,Toperation*B,Toperation*C);
};

#endif