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

#include "OperationType.h"

#define IDX2CGPU(i,j,ld) (((j)*(ld))+(i))

template <class Toperation>
class GpuWorker;

/**
 * @brief Clase estática con métodos estáticos utiles para zonas en las que se utilizan cosas relacionadas con cuda
 * 
 * @tparam Toperation , tipo de la matriz(double,float) 
 */
template <class Toperation>
class MatrixUtilitiesCuda
{
public:
        /**
         * @brief Método crea una matriz de 0s en la gpu del tamaño indicado de forma asíncrona.
         * El dispositivo adecuado debe ser seleccionado antes.
         * 
         * @param rows , filas de la matriz
         * @param columns , columnas de la matriz
         * @param stream , stream con el que realizar la acción
         * @return Toperation* , matriz creada
         */
        static Toperation* cudaMatrixMemoryAllocation(int rows, int columns,cudaStream_t *stream);
        /**
         * @brief Metodo estatico que se encarga de liberar la memoria de la gpu del puntero de la Matriz que se pasa como argumento
         * 
         * @param matrix , Matriz que se va a liberar de la memoria
         */
        static void matrixFree(Toperation *matrix);
        /**
         * @brief Devuelve la id física de la gpu a la que está asociada una gpu lógica
         * 
         * @param gpuRankOperation , id lógico de la gpu en la operación
         * @param gpuSizeSystem , número totales de gpus físicas
         * @return int 
         */
        static int getRealGpuId(int gpuRankOperation,int gpuSizeSystem);
        /**
         * @brief Llamada que ejecuta un kernel de cuda que imprimira la matriz que se quiera de la gpu.
         * El dispositivo adecuado debe de ser seleccionado antes.
         * 
         * @param rows , filas de la matriz
         * @param columns , columnas de la matriz
         * @param matrix , matriz de la gpu a mostrar
         */
        static void cudaPrintOneMatrixCall(int rows,int columns,Toperation* matrix);
        /**
         * @brief Método estático que imprime una matriz con cierto retraso dependiendo del rango de esta.
         * El dispositivo adecuado debe de ser seleccionado antes.
         * 
         * @param gpuRank , rango de la gpu
         * @param rows , filas de la matriz
         * @param columns , columnas de la matriz
         * @param M , matriz
         * @param extraMessage , mensaje adicional por si se quiere mostrar junto la matriz 
         */
        static void cudaDebugMatrixDifferentGpus(int gpuRank, int rows, int columns, Toperation *M, std::string extraMessage);
        /**
         * @brief Método que imprime todas las matrices que tiene un vector de gpuWorkers
         * 
         * @param gpuSizeOperationWorld , número de gpus lógicas que realizan la operación
         * @param gpuSizeSystem , número total de gpus del sistema
         * @param rows , filas de las matrices
         * @param columns , columnas de las matrices
         * @param gpuWorkers , vector de gpuWorkers que cada uno contiene sus matrices
         */
        static void cudaDebugMatricesLocalDifferentGpuWorkers(int gpuSizeOperationWorld,int gpuSizeSystem,int rows, int columns, std::vector<GpuWorker<Toperation>*> gpuWorkers);
        /**
         * @brief Método estático que realiza la multiplicación de de dos matrices mediante cublas y se lo suma a otra matriz. Operación GEMM
         * 
         * @param handler , manejador de cublas
         * @param opt , tipo de operación. MultDouble|MultFloat
         * @param rowsA , filas de la matriz izquierda
         * @param columnsAorRowsB , columnas de la matriz izquierda o filas de la matriz derecha
         * @param columnsB , columnas de la matriz derecha
         * @param A , Matriz izquierda
         * @param B , Matriz derecha
         * @param C , Matriz resultado
         */
        static void matrixCublasMultiplication(cublasHandle_t* handler,OperationType opt,int rowsA, int columnsAorRowsB, int columnsB, Toperation *A, Toperation *B, Toperation *C);
private:
        /**
         * @brief Constructor privado de la clase para que sea estática
         * 
         */
        MatrixUtilitiesCuda();

};