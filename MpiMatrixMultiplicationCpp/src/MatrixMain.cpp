#include "MatrixMain.h"

using namespace std;
template <class Toperation>
MatrixMain<Toperation>::MatrixMain(int rows, int columns)
{
    rowsReal = rows;
    columnsReal = columns;
}

template <class Toperation>
MatrixMain<Toperation>::~MatrixMain()
{
    MatrixUtilities<Toperation>::matrixFree(matrix);
}

template <class Toperation>
void MatrixMain<Toperation>::setMatrix(Toperation* newMatrix)
{
    int i, matrixIndex;
    matrix = MatrixUtilities<Toperation>::matrixMemoryAllocation(rowsUsed, columnsUsed);
    //Bucle de asinacion de la matriz
    for (i = 0; i < rowsReal; i++)
    {
        memcpy(&matrix[i*columnsUsed],&newMatrix[i*columnsReal],columnsReal*sizeof(Toperation));
    }
}

template <class Toperation>
int MatrixMain<Toperation>::getRowsReal()
{
    return rowsReal;
}

template <class Toperation>
int MatrixMain<Toperation>::getColumnsReal()
{
    return columnsReal;
}

template <class Toperation>
int MatrixMain<Toperation>::getRowsUsed()
{
    return rowsUsed;
}

template <class Toperation>
int MatrixMain<Toperation>::getColumnsUsed()
{
    return columnsUsed;
}

template <class Toperation>
bool MatrixMain<Toperation>::getIsDistributed()
{
    return isDistributed;
}

template <class Toperation>
Toperation *MatrixMain<Toperation>::getMatrix()
{
    return matrix;
}

template <class Toperation>
void MatrixMain<Toperation>::setRowsUsed(int rowsUsed)
{
    this->rowsUsed = rowsUsed;
}

template <class Toperation>
void MatrixMain<Toperation>::setColumnsUsed(int columnsUsed)
{
    this->columnsUsed = columnsUsed;
}

template <class Toperation>
void MatrixMain<Toperation>::setIsDistributed(bool isDistributed)
{
    this->isDistributed = isDistributed;
}
template class MatrixMain<double>;
template class MatrixMain<float>;