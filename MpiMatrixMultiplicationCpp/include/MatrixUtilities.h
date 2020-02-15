#ifndef MatrixUtilities_H
#define MatrixUtilities_H

#include <iostream>
#include <cblas.h>
#include <math.h>
#include <unistd.h>
#include <limits>

using namespace std;

class MatrixUtilities
{
    public:
        static void printMatrix(int rows, int columns, double *M);
        static void printMatrixOrMessageForOneCpu(int rows, int columns, double *M,int cpuRank,int cpuRankPrint,string message);
        static void debugMatrixDifferentCpus(int cpurank, int rows, int columns, double *M,string extraMessage);
        static bool canMultiply(int columnsA,int rowsB);
        static int* getMeshAndMatrixSize(int rowsA,int columnsA,int rowsB,int columnsB,int cpuSize );
        static double* matrixCustomAddition(int rows,int columns, double *A, double *B);
        static int matrixCalculateIndex(int columnSize,int rowIndex,int columnIndex);
        static double* matrixBlasMultiplication(int rowsA,int columnsAorRowsB,int columnsB,double* A,double* B,double* C);
        static double* matrixMemoryAllocation(int rows,int columns);
        static void matrixFree(double* matrix);
    private:
        MatrixUtilities();
};
#endif