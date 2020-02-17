#ifndef MpiMultiplicationEnvironment_H
#define MpiMultiplicationEnvironment_H

#include "MatrixUtilities.h"
#include "MpiMatrix.h"

class MpiMultiplicationEnvironment
{
private:
    MPI_Comm commOperation;
    int cpuRank,cpuSize;
    

public:
    MpiMultiplicationEnvironment(int cpuRank,int cpuSize,MPI_Comm commOperation);
    MpiMatrix mpiSumma(MpiMatrix matrixLocalA, MpiMatrix matrixLocalB, int meshRowsSize, int meshColumnsSize);
    void Multiplicacion(int rowsA,int columnsAorRowsB,int columnsB,double* A,double*B,double*C);
};

#endif