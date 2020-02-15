#include "MatrixMain.h"

MatrixMain::MatrixMain(char *fileName)
{
    file.open(fileName);
    file >> rowsReal >> columnsReal;
    //Al ser matriz leida estos valores no se usan
    boundLower = -1;
    boundUpper = -1;
}

MatrixMain::MatrixMain(int rows, int columns, int lowerBound, int upperBound)
{
    rowsReal = rows;
    columnsReal = columns;
    boundLower = lowerBound;
    boundUpper = upperBound;
}

void MatrixMain::fillMatrix(bool isRandom)
{
    int i, j,matrixIndex;
    //Configuracion del generador de numeros por si se genera una matriz random
    random_device rd; 
    mt19937 eng(rd()); 
    uniform_real_distribution<> distr(boundLower, boundUpper);
    matrix=MatrixUtilities::matrixMemoryAllocation(rowsUsed,columnsUsed);
    //Bucle de generacion o lectura de la matriz
    for (i = 0; i < rowsReal; i++)
    {
        for (j = 0; j < columnsReal; j++)
        {
            matrixIndex=MatrixUtilities::matrixCalculateIndex(columnsUsed,i,j);
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

int MatrixMain::getRowsReal()
{
    return rowsReal;
}

int MatrixMain::getColumnsReal()
{
    return columnsReal;
}

int MatrixMain::getRowsUsed()
{
    return rowsUsed;
}

int MatrixMain::getColumnsUsed()
{
    return columnsUsed;
}

double *MatrixMain::getMatrix()
{
    return matrix;
}

void MatrixMain::setRowsUsed(int rowsUsed)
{
    this->rowsUsed=rowsUsed;
}

void MatrixMain::setColumnsUsed(int columnsUsed)
{
    this->columnsUsed=columnsUsed;
}