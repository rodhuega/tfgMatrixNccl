#ifndef MatrixUtilities_H
#define MatrixUtilities_H

#include <iostream>

using namespace std;

class MatrixUtilities
{
    public:
        static void PrintDoublePointerMatrix(int rows, int columns, double **M);
        static void PrintOnePointerMatrix(int rows, int columns, double *M);
        void matrixMultiplication(int N, double *A, double *B, double *C);
    private:
        MatrixUtilities();
};
#endif