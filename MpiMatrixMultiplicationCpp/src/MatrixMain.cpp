#include "MatrixMain.h"

using namespace std;

template <class Toperation>
MatrixMain<Toperation>::MatrixMain(const char *fileName)
{
    file.open(fileName);
    file >> rowsReal >> columnsReal;
    //Al ser matriz leida de un fichero estos valores no se usan
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
MatrixMain<Toperation>::~MatrixMain()
{
    MatrixUtilities<Toperation>::matrixFree(matrix);
}

template <class Toperation>
void MatrixMain<Toperation>::fillMatrix(int fillType, Toperation* matrixFromMemory)
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
            if (fillType==random)
            {
                matrix[matrixIndex] = distr(eng) ;
            }
            else if(fillType==fromFile)
            {
                file >> matrix[matrixIndex];
            }else
            {

            }
        }
    }
    if(fillType==fromFile)
    {
        file.close();
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