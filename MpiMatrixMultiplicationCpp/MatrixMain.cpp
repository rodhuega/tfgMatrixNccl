#include "MatrixMain.h"

MatrixMain::MatrixMain(char *fileName)
{
    file.open(fileName);
    file >> rowsReal >> columnsReal;
    fillMatrix(false);
    file.close();
}

MatrixMain::MatrixMain(int rows, int columns,int lowerBound, int upperBound)
{
    rowsReal = rows;
    columnsReal = columns;
    boundLower=lowerBound;
    boundUpper=upperBound;
    fillMatrix(true);
}

void MatrixMain::fillMatrix(bool isRandom)
{
    int i, j;
    rowsUsed = rowsReal % 2 ? rowsReal + 1 : rowsReal;
    columnsUsed = columnsReal % 2 ? columnsReal + 1 : columnsReal;
    extendedRow = !(rowsUsed == rowsReal);
    extendedColumn = !(columnsUsed == columnsReal);
    matrix = new double *[rowsUsed];
    for (i = 0; i < rowsReal; i++)
    {
        matrix[i] = new double[rowsUsed * columnsUsed];
        for (j = 0; j < columnsReal; j++)
        {
            if (isRandom)
            {
                matrix[i][j] = boundLower + rand() % (boundUpper +1 - boundLower);
            }
            else
            {
                file >> matrix[i][j];
            }
        }
        if (extendedColumn)
        {
            matrix[i][j] = 0.0;
        }
    }
    if (extendedRow)
    {
        matrix[i]= (double*)calloc(rowsUsed*columnsUsed,sizeof(double));
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

double **MatrixMain::getMatrix()
{
    return matrix;
}