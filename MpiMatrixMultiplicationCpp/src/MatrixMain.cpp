#include "MatrixMain.h"

using namespace std;

template <class Toperation>
MatrixMain<Toperation>::MatrixMain(){}

template <class Toperation>
MatrixMain<Toperation>::MatrixMain(char *fileName)
{
    file.open(fileName);
    file >> rowsReal >> columnsReal;
    //Al ser matriz leida estos valores no se usan
    boundLower = -1;
    boundUpper = -1;
}

template <class Toperation>
MatrixMain<Toperation>::MatrixMain(int rows, int columns, int lowerBound, int upperBound)
{
    rowsReal = rows;
    columnsReal = columns;
    boundLower = lowerBound;
    boundUpper = upperBound;
}

template <class Toperation>
void MatrixMain<Toperation>::fillMatrix(bool isRandom)
{
    int i, j,matrixIndex;
    //Configuracion del generador de numeros por si se genera una matriz random
    random_device rd; 
    mt19937 eng(rd()); 
    uniform_real_distribution<> distr(boundLower, boundUpper);
    matrix=MatrixUtilities<Toperation>::matrixMemoryAllocation(rowsUsed,columnsUsed);
    //Bucle de generacion o lectura de la matriz
    for (i = 0; i < rowsReal; i++)
    {
        for (j = 0; j < columnsReal; j++)
        {
            matrixIndex=MatrixUtilities<Toperation>::matrixCalculateIndex(columnsUsed,i,j);
            if (isRandom)
            {
                matrix[matrixIndex] = distr(eng) ;
            }
            else
            {
                file >> matrix[matrixIndex];
            }
        }
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
Toperation *MatrixMain<Toperation>::getMatrix()
{
    return matrix;
}

template <class Toperation>
void MatrixMain<Toperation>::setRowsUsed(int rowsUsed)
{
    this->rowsUsed=rowsUsed;
}

template <class Toperation>
void MatrixMain<Toperation>::setColumnsUsed(int columnsUsed)
{
    this->columnsUsed=columnsUsed;
}
template class MatrixMain<double>;
template class MatrixMain<float>;
template class MatrixMain<int>;