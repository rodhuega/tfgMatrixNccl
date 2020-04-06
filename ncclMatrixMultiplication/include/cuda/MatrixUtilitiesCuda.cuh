#pragma once

#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <vector>

#include "cuda_runtime.h"
#include <cublas_v2.h>
#include "nccl.h"

#include "GpuWorker.cuh"

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

template <class Toperation>
class GpuWorker;
template <class Toperation>
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

        static Toperation* cudaMatrixMemoryAllocation(int rows, int columns,cudaStream_t *stream);

        static int getRealGpuId(int gpuRankOperation,int gpuSizeSystem);

        static void cudaPrintOneMatrixCall(int rows,int columns,Toperation* matrix);

        static void cudaDebugMatrixDifferentGpus(int gpuRank, int rows, int columns, Toperation *M, std::string extraMessage);

        static void cudaDebugMatricesLocalDifferentGpuWorkers(int cpuSize, int rows, int columns, std::vector<GpuWorker<Toperation>*> gpuWorkers);
};