#ifndef MatrixUtilitiesCuda_H
#define MatrixUtilitiesCuda_H

#include <stdio.h>

#include "cuda_runtime.h"
#include <cublas_v2.h>
#include "nccl.h"

#define IDX2C(i,j,ld) (((j)*(ld))+(i))



class MatrixUtilitiesCuda
{
public:
        /**
        * @brief Metodo estatico que calcula la posicion del puntero unidimensional de un elemento de una matriz
        * 
        * @param rowSize , tamaño de las filas de la matriz
        * @param columnSize , tamaño de las columnas de la matriz
        * @param rowIndex , fila del elemento al que se quiere acceder
        * @param columnIndex , columna del elemento al que se quiere acceder
        * @return int 
        */
        static int matrixCalculateIndex(int rowSize,int columnSize, int rowIndex, int columnIndex);

        static void cudaPrintMatrixCall(int rows,int columns,double* matrix);
};
#endif