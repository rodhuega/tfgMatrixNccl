#include "MatrixMain.h"
#include "MatrixUtilities.h"

MatrixMain::MatrixMain(char *fileName)
{
    file.open(fileName);
    file >> rowsReal >> columnsReal;
    fillMatrix(false);
    file.close();
}

MatrixMain::MatrixMain(int rows, int columns, int lowerBound, int upperBound)
{
    rowsReal = rows;
    columnsReal = columns;
    boundLower = lowerBound;
    boundUpper = upperBound;
    fillMatrix(true);
}

void MatrixMain::fillMatrix(bool isRandom)
{
    int i, j,matrixIndex;
    //Configuracion del generador de numeros por si se genera una matriz random
    random_device rd; 
    mt19937 eng(rd()); 
    uniform_real_distribution<> distr(boundLower, boundUpper);
    //Creo que me puedo cargar los booleanos y ademas necesitare alguna formula para extender tantos 0 como sea necesario
    //para asi poder multiplicar matrices que no sean iguales y que no sean cuadradas
    rowsUsed = rowsReal % 2 ? rowsReal + 1 : rowsReal;
    columnsUsed = columnsReal % 2 ? columnsReal + 1 : columnsReal;
    //WIP: INICIALIZACION no me fio de los calloc, aun asi creo que me puedo cargar los comentarios
    matrix=MatrixUtilities::matrixMemoryAllocation(rowsReal,columnsReal);
    for (i = 0; i < rowsReal; i++)
    {
        for (j = 0; j < columnsReal; j++)
        {
            matrixIndex=MatrixUtilities::matrixCalculateIndex(rowsReal,i,j);
            if (isRandom)
            {
                matrix[matrixIndex] = distr(eng) ;
            }
            else
            {
                file >> matrix[matrixIndex];
            }
        }
        // if (rowsUsed!=rowsReal)
        // {
        //     matrix[matrixIndex] = 0.0;
        // }
    }
    // if (columnsUsed!=columnsReal)
    // {
    //     matrix[i] = (double *)calloc(rowsUsed * columnsUsed, sizeof(double));
    // }
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