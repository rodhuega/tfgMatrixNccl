#include "MatrixUtilities.h"
#include <cblas.h>
#include <unistd.h>


void MatrixUtilities::printMatrix(int rows, int columns, double *M)
{
    int i, j,matrixIndex;
    for (i = 0; i < rows; ++i)
    {
        for (j = 0; j < columns; ++j)
        {
            matrixIndex=matrixCalculateIndex(rows,i,j);
            cout << M[matrixIndex] << "\t";
        }
        cout << endl;
    }
}

void MatrixUtilities::printMatrixOrMessageForOneCpu(int rows, int columns, double *M,int cpuRank,int cpuRankPrint,string message)
{
    if(cpuRank==cpuRankPrint)
    {
        cout<<message<<endl;
        if (M!=NULL)
        {
            printMatrix(rows,columns,M);
        }
    }
}

void MatrixUtilities::debugMatrixDifferentCpus(int cpuRank, int rows, int columns, double *M,string extraMessage)
{
    usleep(cpuRank*1000);
    cout<<"Parte del proceso: "<<cpuRank<<" "<<extraMessage<<endl;
    MatrixUtilities::printMatrix(rows,columns,M);
}


double* MatrixUtilities::matrixMemoryAllocation(int rows,int columns)
{
    double* matrix=(double *)calloc(rows * columns, sizeof(double));
    return matrix;
}

void MatrixUtilities::matrixFree(double* matrix)
{
    free(matrix);
}

double* MatrixUtilities::matrixCustomAddition(int rows,int columns, double *A, double *B) 
{
    int i,j,matrixIndex;
    double* res = matrixMemoryAllocation(rows,columns);
    for (i = 0; i < rows; ++i) {
        for (j = 0; j < columns; ++j) {
            matrixIndex=matrixCalculateIndex(rows,i,j);
            res[matrixIndex] = A[matrixIndex] + B[matrixIndex];
        }
    }
    return res;
}

/**
 * @brief Calcula el indice unidimensional de la matriz
 * 
 * @param rowSize 
 * @param rowIndex 
 * @param columnIndex 
 * @return int 
 */
int MatrixUtilities::matrixCalculateIndex(int rowSize,int rowIndex,int columnIndex)
{
    return rowSize*rowIndex+columnIndex;
}

double* MatrixUtilities::matrixBlasMultiplication(int rowsA,int columnsAorRowsB,int columnsB,double* A,double* B,double* C)
{
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rowsA, columnsB,columnsAorRowsB, 1.0, A, rowsA, B, columnsAorRowsB, 1.0, C, rowsA);
    return C;
}