#pragma once

#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <fstream>
#include <math.h>
#include <limits>
#include <algorithm>
#include <tuple>
#include <iterator>
#include "mkl.h"

#include "cuda_runtime.h"
#include <cublas_v2.h>
#include <curand.h>
#include "nccl.h"

#include "GpuWorker.cuh"
#include "ErrorCheckingCuda.cuh"

#include "OperationType.h"
#include "OperationProperties.h"

#define IDX2C(i, j, ld) (((j) * (ld)) + (i))

template <class Toperation>
class GpuWorker;

/**
 * @brief Clase estática con métodos estáticos útiles para zonas en las que se utilizan cosas relacionadas con cuda
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
        static Toperation *cudaMatrixMemoryAllocationGPU(int rows, int columns, cudaStream_t *stream);
        /**
         * @brief Método estático que se encarga de liberar la memoria de la gpu del puntero de la Matriz que se pasa como argumento
         * 
         * @param matrix , Matriz que se va a liberar de la memoria
         */
        static void matrixFreeGPU(Toperation *matrix);
        /**
         * @brief Método estático que devuelve la id física de la gpu a la que está asociada una gpu lógica
         * 
         * @param gpuRankOperation , id lógico de la gpu en la operación
         * @param gpuSizeSystem , número totales de gpus físicas
         * @return int 
         */
        static int getRealGpuId(int gpuRankOperation, int gpuSizeSystem);
        /**
         * @brief Llamada que ejecuta un kernel de cuda que imprimirá la matriz que se quiera de la gpu.
         * El dispositivo adecuado debe de ser seleccionado antes.
         * 
         * @param rows , filas de la matriz
         * @param columns , columnas de la matriz
         * @param matrix , matriz de la gpu a mostrar
         * @param opt , tipo de operación. MultDouble|MultFloat
         */
        static void cudaPrintOneMatrixCall(int rows, int columns, Toperation *matrix, OperationType opt);
        /**
         * @brief Método estático que imprime una matriz con cierto retraso dependiendo del rango de esta.
         * El dispositivo adecuado debe de ser seleccionado antes.
         * 
         * @param gpuRank , rango de la gpu
         * @param rows , filas de la matriz
         * @param columns , columnas de la matriz
         * @param M , matriz
         * @param extraMessage , mensaje adicional por si se quiere mostrar junto la matriz 
         * @param opt , tipo de operación. MultDouble|MultFloat
         */
        static void cudaDebugMatrixDifferentGpus(int gpuRank, int rows, int columns, Toperation *M, std::string extraMessage, OperationType opt);
        /**
         * @brief Método que imprime todas las matrices que tiene un vector de gpuWorkers
         * 
         * @param gpuSizeOperationWorld , número de gpus lógicas que realizan la operación
         * @param gpuSizeSystem , número total de gpus del sistema
         * @param rows , filas de las matrices
         * @param columns , columnas de las matrices
         * @param gpuWorkers , vector de gpuWorkers que cada uno contiene sus matrices
         * @param opt , tipo de operación. MultDouble|MultFloat
         */
        static void cudaDebugMatricesLocalDifferentGpuWorkers(int gpuSizeOperationWorld, int gpuSizeSystem, int rows, int columns, std::vector<GpuWorker<Toperation> *> gpuWorkers, OperationType opt);
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
         * @param alpha , número por el que se multiplica A(matriz izquierda)
         * @param beta , número por el que se suma B(matriz derecha)
         */
        static void matrixCublasMultiplication(cublasHandle_t *handler, OperationType opt, int rowsA, int columnsAorRowsB, int columnsB, Toperation *A, Toperation *B, Toperation *C, Toperation alpha, Toperation beta);
        /**
         * @brief Método estático que realiza la operacion axpy mediante cublas para una matriz
         * 
         * @param handler , manejador de cublas
         * @param opt , tipo de operación. MultDouble|MultFloat
         * @param numberOfElementsToOperate , número total de elementos sobre los que hay que operar.
         * @param X , Matrix X
         * @param Y , Matrix Y
         * @param alpha , escalar
         * @param strideX , separación entre los elementos de X
         * @param strideY , separación entre los elementos de Y
         */
        static void axpyCublas(cublasHandle_t *handler, OperationType opt, int numberOfElementsToOperate, Toperation *X, Toperation *Y, Toperation alpha, Toperation strideX, Toperation strideY);
        /**
         * @brief Método estático que realiza la operacion scalar mediante cublas para una matriz
         * 
         * @param handler , manejador de cublas
         * @param opt , tipo de operación. MultDouble|MultFloat
         * @param rows , filas de la matriz
         * @param columns , columnas de la matriz
         * @param X , Matrix X
         * @param alpha , escalar
         * @param strideX , separación entre los elementos de X
         */
        static void scalarCublas(cublasHandle_t *handler, OperationType opt, int rows, int columns, Toperation *X, Toperation alpha, Toperation strideX);
        /**
         * @brief Método estático que realiza la suma de un vector mediante cublas
         * 
         * @param handler , manejador de cublas
         * @param opt , tipo de operación. MultDouble|MultFloat
         * @param n , número de elementos del vector
         * @param strideX , separación entre los elementos de X
         * @param X , puntero del vector a sumar
         * @param result , puntero resultado (Puede ser host o device)
         */
        static void sumCublas(cublasHandle_t *handler, OperationType opt, int n,int strideX, Toperation *X, Toperation *result);
        /**
         * @brief Genera una matriz aleatoria entre 0 y 1 mediante curand
         * 
         * @param rows , filas de la matriz
         * @param columns , columnas de la matriz
         * @param opt , tipo de operación. MultDouble|MultFloat
         * @return Toperation* , puntero a la matriz aleatoria
         */
        static Toperation *GenerateRandomMatrixGPU(int rows, int columns, OperationType opt);
        /**
         * @brief Método estático imprime por pantalla una matriz
         * 
         * @param rows , Filas de la matriz
         * @param columns , Columnas de la matriz
         * @param M , Matriz a mostrar por pantalla
         */
        static void printMatrix(int rows, int columns, Toperation *M);
        /**
         * @brief Método estático devuelve el error relativo de dos matrices mediante la norma de Frobenius
         * 
         * @param A , Primera matriz a comparar
         * @param B , Segunda matriz a comparar
         * @param rows , Filas de las matrices
         * @param columns , Columnas de las matrices
         * @return std::vector<std::tuple<int,int>> , vector de tuplas con las posiciones donde no coinciden
         */
        static double checkEqualityOfMatrices(Toperation *A, Toperation *B, int rows, int columns);
        /**
         * @brief Método estático que comprueba si dos matrices se pueden multiplicar entre sí.
         * 
         * @param columnsA , columnas de la matriz A
         * @param rowsB , filas de la matriz B
         * @return true 
         * @return false 
         */
        static bool canMultiply(int columnsA, int rowsB);
        /**
         * @brief Método estático que calcula el tamaño de la malla y de las matrices operacionales para realizar el cálculo
         * 
         * @param rowsA , Filas de la matriz B
         * @param columnsA , columnas de la matriz A
         * @param rowsB , filas de la matriz B
         * @param columnsB , columnas de la matriz B
         * @param cpuSize número de procesadores disponibles para realizar el cálculo
         * @return OperationProperties 
         */
        static OperationProperties getMeshAndMatrixSize(int rowsA, int columnsA, int rowsB, int columnsB, int cpuSize);
        /**
         * @brief Método estático que calcula las propiedades de la distribución de una matriz si ya hay una distribuida
         * 
         * @param rowsA , Filas de la matriz B
         * @param columnsA , columnas de la matriz A
         * @param rowsB , filas de la matriz B
         * @param columnsB , columnas de la matriz B
         * @param meshRowSize , tamaño de las filas de la malla
         * @param meshColumnSize , tamaño de las columnas de la malla
         * @param isAAlreadyDistributed , si la matriz distribuida es la A(la de la izquierda)
         * @return OperationProperties 
         */
        static OperationProperties getMeshAndMatrixSizeFromOneDistributedMatrix(int rowsA, int columnsA, int rowsB, int columnsB, int meshRowSize, int meshColumnSize, bool isAAlreadyDistributed);
        /**
         * @brief Método estático que calcula la posición del puntero unidimensional de un elemento de una matriz
         * 
         * @param rowSize , tamaño de las filas de la matriz
         * @param columnSize , tamaño de las columnas de la matriz
         * @param rowIndex , fila del elemento al que se quiere acceder
         * @param columnIndex , columna del elemento al que se quiere acceder
         * @return unsigned long long 
         */
        static unsigned long long matrixCalculateIndex(int rowSize, int columnSize, int rowIndex, int columnIndex);
        /**
         * @brief Método estático que reserva memoria para una matriz
         * 
         * @param rows , filas de la matriz
         * @param columns , columnas de la matriz
         * @return Toperation* , Puntero de la matriz que se ha reservado en memoria
         */
        static Toperation *matrixMemoryAllocationCPU(int rows, int columns);
        /**
         * @brief Método estático que se encarga de liberar la memoria del host(cpu) del puntero de la Matriz que se pasa como argumento
         * 
         * @param matrix , Matriz que se va a liberar de la memoria
         */
        static void matrixFreeCPU(Toperation *matrix);
        /**
         * @brief 
         * 
         * @param isRandom , Indica si la matriz se va a generar de forma aleatoria
         * @param fileName , Fichero del que se va a leer la matriz en el caso de que asi sea. Puede ser cualquier valor en caso de que no sea así
         * @param rows , filas de la matriz
         * @param columns , columnas de la matriz
         * @param boundLower , límite inferior en el caso de que sea una matriz random. Puede ser cualquier valor en caso de que no sea así
         * @param boundUpper , límite superior en el caso de que sea una matriz random. Puede ser cualquier valor en caso de que no sea así
         * @return Toperation* 
         */
        static Toperation *ReadOrGenerateRandomMatrix(bool isRandom, const char *fileName, int &rows, int &columns, int boundLower, int boundUpper);
        /**
         * @brief Método estático que realiza la operacion axpy mediante cblas para una matriz
         * 
         * @param opt , tipo de operación. MultDouble|MultFloat
         * @param numberOfElementsToOperate , número total de elementos sobre los que hay que operar.
         * @param X , Matrix X
         * @param Y , Matrix Y
         * @param alpha , escalar
         * @param strideX , separación entre los elementos de X
         * @param strideY , separación entre los elementos de Y
         */
        static void axpyBlas(OperationType opt, int numberOfElementsToOperate, Toperation *X, Toperation *Y, Toperation alpha, Toperation strideX, Toperation strideY);
        /**
         * @brief  Método estático que devuelve el valor máximo de un array gracias a cblas_i?max
         * 
         * @param opt , tipo de operación. MultDouble|MultFloat
         * @param numberOfElementsToOperate , número total de elementos sobre los que hay que operar.
         * @param X , array el cual va a ser analizado
         * @param strideX , separación entre los elementos del array
         * @return Toperation 
         */
        static Toperation maximumBlas(OperationType opt, int numberOfElementsToOperate, Toperation *X, Toperation strideX);
        /**
         * @brief  Método estático que devuelve el valor del indice máximo del array en indexMax. Tdoo mediante cublas.
         * 
         * @param handler , manejador de cublas
         * @param opt , tipo de operación. MultDouble|MultFloat
         * @param numberOfElementsToOperate , número total de elementos sobre los que hay que operar.
         * @param X , array el cual va a ser analizado
         * @param strideX , separación entre los elementos del array
         * @param indexMax, puntero de donde se almacenara el índice máximo
         */
        static void maximumCublas(cublasHandle_t *handler,OperationType opt, int numberOfElementsToOperate, Toperation *X, int strideX,int *indexMax);

private:
        /**
         * @brief Constructor privado de la clase para que sea estática
         * 
         */
        MatrixUtilitiesCuda();
        /**
         * @brief Método privado que mira que calcula las propiedades de la operación de forma individual para ese número de cpus en cada lado de la malla
         * 
         * @tparam Toperation , Tipo basico de la operación de la matriz
         * @param rowsA , filas de la matriz A
         * @param columnsAorRowsB , filas de B o columnas de A
         * @param columnsB , columnas de B
         * @param nCpusMesh1 , cpus en una dimensión de la malla
         * @param nCpusMesh2 , cpus en otra dimensión de la malla
         * @param isMeshRow ,si es verdadero la nCpusMesh1 va para meshRowSize, si es falso para meshColumnSize
         * @return OperationProperties , propiedades de la operación
         */
        static OperationProperties calculateNonEqualMesh(int rowsA, int columnsAorRowsB, int columnsB, int nCpusMesh1, int nCpusMesh2, bool isMeshRow);
};