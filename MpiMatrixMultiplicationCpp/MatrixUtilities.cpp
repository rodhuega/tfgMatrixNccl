#include "MatrixUtilities.h"

void MatrixUtilities::PrintDoublePointerMatrix(int rows, int columns, double **M)
{
    int i, j;
    for (i = 0; i < rows; ++i)
    {
        for (j = 0; j < columns; ++j)
        {
            cout << M[i][j] << "\t";
        }
        cout << endl;
    }
}

void MatrixUtilities::PrintOnePointerMatrix(int rows, int columns, double *M)
{
    int i, j;
    for (i = 0; i < rows; ++i)
    {
        for (j = 0; j < columns; ++j)
        {
            cout << M[i * rows + j] << "\t";
        }
        cout << endl;
    }
}

void MatrixUtilities::matrixMultiplication(int N, double *A, double *B, double *C)
{
    int i, j, k, sum;
    for (i = 0; i < N; i++)
    {
        sum = 0;
        for (j = 0; j < N; j++)
        {
            sum = 0;
            for (k = 0; k < N; k++)
            {
                sum += A[i * N + j] * B[i + N * j];
            }
            C[i*N+j]= sum;
        }
    }
}