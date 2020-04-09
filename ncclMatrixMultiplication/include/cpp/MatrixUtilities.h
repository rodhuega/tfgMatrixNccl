#pragma once

#include <iostream>
#include <random>
#include <fstream>
#include <string>
#include <cblas.h>
#include <math.h>
#include <unistd.h>
#include <limits>
#include <algorithm>
#include <vector>
#include <tuple>
#include <iterator>

#include <OperationProperties.h>
#include "OperationType.h"


#define IDX2C(i, j, ld) (((j) * (ld)) + (i))
/**
 * @brief Clase estática con métodos estáticos útiles en todos los lugares del programa
 * 
 * @tparam Toperation , tipo de la matriz(double,float) 
 */
template <class Toperation>
class MatrixUtilities
{
public:
     /**
         * @brief Método estático imprime por pantalla una matriz
         * 
         * @param rows , Filas de la matriz
         * @param columns , Columnas de la matriz
         * @param M , Matriz a mostrar por pantalla
         */
     static void printMatrix(int rows, int columns, Toperation *M);
     /**
         * @brief Método estático que imprime una matriz o un mensaje o ambos para un único proceso
         * 
         * @param rows , filas de la matriz
         * @param columns , columnas de la matriz
         * @param M , matriz a mostrar por pantalla, en caso de ser NULL no se mostrara ninguna matriz
         * @param cpuRank , Id de la cpu
         * @param cpuRankPrint , Id de la cpu que se desea que imprima la información por la pantalla
         * @param message , mensaje a mostrar por pantalla, puede ser vacio("") en caso de que no se quiera mostar nada
         */
     static void printMatrixOrMessageForOneCpu(int rows, int columns, Toperation *M, int cpuRank, int cpuRankPrint, std::string message);
     /**
         * @brief Método estático que imprime la matriz que tiene cada proceso con un retraso de su ID * 1000 ms. Pensada para ver los bloques de cada proceso
         * 
         * @param cpurank , Id de la cpu
         * @param rows , filas de la matriz
         * @param columns , columnas de la matriz
         * @param M , Matriz a mostrar
         * @param extraMessage , mensaje a mostrar por pantalla, puede ser vacio("") en caso de que no se quiera mostar nada
         */
     static void debugMatrixDifferentCpus(int cpurank, int rows, int columns, Toperation *M, std::string extraMessage);
     /**
         * @brief Método estático que imprime las distintas matrices locales que tiene cada proceso con un retraso de su ID * 1000 ms. Pensada para ver los diferente bloques que pueda tener cada proceso
         * 
         * @param cpurank , Id de la cpu
         * @param cpuSize , Número de procesadores que van a realizar la operacion matemática
         * @param rows , filas de la matriz
         * @param columns , columnas de la matriz
         * @param vector<Toperation*>M , vector que contiene las matrices de cada proceso que hay que mostrar
         * @param extraMessage , mensaje a mostrar por pantalla, puede ser vacio("") en caso de que no se quiera mostar nada
         */
     static void debugMatricesLocalDifferentCpus(int cpurank, int cpuSize, int rows, int columns, std::vector<Toperation *> M, std::string extraMessage);
     /**
         * @brief Método estático que mira en que posiciones dos matrices no son iguales con una diferencia de 0.000001 para cada elemento
         * 
         * @param A , Primera matriz a comparar
         * @param B , Segunda matriz a comparar
         * @param rows , Filas de las matrices
         * @param columns , Columnas de las matrices
         * @return std::vector<std::tuple<int,int>> , vector de tuplas con las posiciones donde no coinciden
         */
     static std::vector<std::tuple<int, int>> checkEqualityOfMatrices(Toperation *A, Toperation *B, int rows, int columns);
     /**
         * @brief Metodo estatico que imprime por pantalla las posiciones donde las dos matrices no tienen los mismos elementos
         * 
         * @param errors , un vector de tuplas con las posiciones donde no coinciden
         * @param printDetailed , Indica si se van a pintar las posiciones donde no es igual la matriz en caso de que no sea igual
         */
     static void printErrorEqualityMatricesPosition(std::vector<std::tuple<int, int>> errors, bool printDetailed);
     /**
         * @brief Método estático que comprueba si dos matrices se pueden multiplicar entre si
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
         * @brief Método estático que calcula la posición del puntero unidimensional de un elemento de una matriz
         * 
         * @param rowSize , tamaño de las filas de la matriz
         * @param columnSize , tamaño de las columnas de la matriz
         * @param rowIndex , fila del elemento al que se quiere acceder
         * @param columnIndex , columna del elemento al que se quiere acceder
         * @return int 
         */
     static int matrixCalculateIndex(int rowSize, int columnSize, int rowIndex, int columnIndex);
     /**
         * @brief Método estático que reserva memoria para una matriz
         * 
         * @param rows , filas de la matriz
         * @param columns , columnas de la matriz
         * @return Toperation* , Puntero de la matriz que se ha reservado en memoria
         */
     static Toperation *matrixMemoryAllocation(int rows, int columns);
     /**
      * @brief Método estático que se encarga de liberar la memoria del host(cpu) del puntero de la Matriz que se pasa como argumento
      * 
      * @param matrix , Matriz que se va a liberar de la memoria
      */
     static void matrixFree(Toperation *matrix);
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
         * @brief Método estático que multiplica 2 matrices A y B y suma el resultado en C mediante la líbreria cblas Operacion C+=A*B
         * 
         * @param opt , tipo de operación. MultDouble|MultFloat
         * @param rowsA , Filas de A
         * @param columnsAorRowsB , Filas de B o columnas de A
         * @param columnsB , Columnas de B
         * @param A , Matriz A
         * @param B , Matriz B
         * @param C , Matriz a la cual es le va a sumar el resultado de A*B
         */
     static void matrixBlasMultiplication(OperationType opt,int rowsA, int columnsAorRowsB, int columnsB, Toperation *A, Toperation *B, Toperation *C);

private:
     /**
      * @brief Constructor privado de la clase para que sea estática
      * 
      */
     MatrixUtilities();
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