#ifndef MatrixUtilities_H
#define MatrixUtilities_H

#include <iostream>

using namespace std;

class MatrixUtilities
{
    public:
        static void printOnePointerMatrix(int rows, int columns, double *M);
        static void debugMatrixDifferentCpus(int cpurank, int rows, int columns, double *M);
        static double* matrixCustomAddition(int rows,int columns, double *A, double *B);
        static int matrixCalculateIndex(int rowSize,int rowIndex,int columnIndex);
        static double* matrixBlasMultiplication(int rowsA,int columnsAorRowsB,int columnsB,double* A,double* B);
        static double* matrixMemoryAllocation(int rows,int columns);
        static void matrixFree(double* matrix);
    private:
        MatrixUtilities();
};
#endif