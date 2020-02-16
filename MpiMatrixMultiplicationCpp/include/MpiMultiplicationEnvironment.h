#ifndef MpiMultiplicationEnvironment_H
#define MpiMultiplicationEnvironment_H

#include "MatrixUtilities.h"
#include "MpiMatrix.h"

class MpiMultiplicationEnvironment
{
private:
    int cpuRank,cpuSize;
    

public:
    MpiMultiplicationEnvironment(int cpuRank,int cpuSize);
    MpiMatrix mpiSumma(MpiMatrix matrixLocalA, MpiMatrix matrixLocalB, int meshRowsSize, int meshColumnsSize);
};

#endif