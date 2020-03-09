#include "MatrixMain.h"

using namespace std;
template <class Toperation>
MatrixMain<Toperation>::MatrixMain(int rows, int columns)
{
    isMatrixGlobalHere=false;
    isDistributed=false;
    rowsReal = rows;
    columnsReal = columns;
}

template <class Toperation>
MatrixMain<Toperation>::~MatrixMain()
{
    MatrixUtilities<Toperation>::matrixFree(matrixGlobal);
}

template <class Toperation>
void MatrixMain<Toperation>::setMatrix(Toperation* matrixGlobalNew)
{
    isMatrixGlobalHere=true;
    matrixGlobal=matrixGlobalNew;
}

template <class Toperation>
void MatrixMain<Toperation>::setMpiMatrix(MpiMatrix<Toperation> *matrixLocal)
{
    isDistributed=true;
    this->matrixLocal=matrixLocal;
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
bool MatrixMain<Toperation>::getIsMatrixGlobalHere()
{
    return isMatrixGlobalHere;
}

template <class Toperation>
MpiMatrix<Toperation>* MatrixMain<Toperation>::getMpiMatrix()
{
    return matrixLocal;
}

template <class Toperation>
Toperation *MatrixMain<Toperation>::getMatrix()
{
    return matrixGlobal;
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

template <class Toperation>
void MatrixMain<Toperation>::setIsMatrixGlobalHere(bool isMatrixGlobalHere)
{
    this->isMatrixGlobalHere = isMatrixGlobalHere;
}

template <class Toperation>
void MatrixMain<Toperation>::eraseMatrixGlobal()
{
    MatrixUtilities<Toperation>::matrixFree(matrixGlobal);
}

template class MatrixMain<double>;
template class MatrixMain<float>;