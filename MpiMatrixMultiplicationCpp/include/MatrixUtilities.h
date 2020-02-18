#ifndef MatrixUtilities_H
#define MatrixUtilities_H

#include <iostream>
#include <string>
#include <cblas.h>
#include <math.h>
#include <unistd.h>
#include <limits>
#include <algorithm>
#include <vector>
#include <iterator>
#include <OperationProperties.h>

template <class Toperation>
class MatrixUtilities
{
    public:
        static void printMatrix(int rows, int columns, Toperation *M);
        static void printMatrixOrMessageForOneCpu(int rows, int columns, Toperation *M,int cpuRank,int cpuRankPrint,std::string message);
        static void debugMatrixDifferentCpus(int cpurank, int rows, int columns, Toperation *M,std::string extraMessage);
        static Toperation* getMatrixWithoutZeros(int rowsReal,int columnsUsed, int columnsReal,Toperation* matrix);
        static bool canMultiply(int columnsA,int rowsB);
        static OperationProperties getMeshAndMatrixSize(int rowsA,int columnsA,int rowsB,int columnsB,int cpuSize );
        static Toperation* matrixCustomAddition(int rows,int columns, Toperation *A, Toperation *B);
        static int matrixCalculateIndex(int columnSize,int rowIndex,int columnIndex);
        static Toperation* matrixBlasMultiplication(int rowsA,int columnsAorRowsB,int columnsB,Toperation* A,Toperation* B,Toperation* C);
        static Toperation* matrixMemoryAllocation(int rows,int columns);
        static void matrixFree(Toperation* matrix);
    private:
        MatrixUtilities();
        static OperationProperties calculateNonEqualMesh(int rowsA, int columnsAorRowsB, int columnsB, int nCpusMesh,  int cpuSize,bool isMeshRow);
};
#endif