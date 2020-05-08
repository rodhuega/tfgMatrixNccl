#ifndef MatrixUtilities_H
#define MatrixUtilities_H

#include <iostream>
#include <random>
#include <fstream>
#include <string>
// #include <cblas.h>
#include "mkl.h"
#include <math.h>
#include <unistd.h>
#include <limits>
#include <algorithm>
#include <vector>
#include <tuple>
#include <iterator>
#include <OperationProperties.h>
#include "omp.h"

/**
 * @brief Clase estatica con metodos estaticos utiles en todos los lugares del programa
 * 
 * @tparam Toperation , tipo de la matriz(double,float) 
 */
template <class Toperation>
class MatrixUtilities
{
public:
    /**
         * @brief Metodo estatico imprime por pantalla una matriz
         * 
         * @param rows , Filas de la matriz
         * @param columns , Columnas de la matriz
         * @param M , Matriz a mostrar por pantalla
         */
    static void printMatrix(int rows, int columns, Toperation *M);
    /**
         * @brief Metodo estatico que imprime una matriz o un mensaje o ambos para un unico proceso
         * 
         * @param rows , filas de la matriz
         * @param columns , columnas de la matriz
         * @param M , matriz a mostrar por pantalla, en caso de ser NULL no se mostrara ninguna matriz
         * @param cpuRank , Id de la cpu
         * @param cpuRankPrint , Id de la cpu que se desea que imprima la informacion por la pantalla
         * @param message , mensaje a mostrar por pantalla, puede ser vacio("") en caso de que no se quiera mostar nada
         */
    static void printMatrixOrMessageForOneCpu(int rows, int columns, Toperation *M, int cpuRank, int cpuRankPrint, std::string message);
    /**
         * @brief Metodo estatico que imprime la matriz que tiene cada proceso con un retraso de su ID * 1000 ms. Pensada para ver los bloques de cada proceso
         * 
         * @param cpurank , Id de la cpu
         * @param rows , filas de la matriz
         * @param columns , columnas de la matriz
         * @param M , Matriz a mostrar
         * @param extraMessage , mensaje a mostrar por pantalla, puede ser vacio("") en caso de que no se quiera mostar nada
         */
    static void debugMatrixDifferentCpus(int cpurank, int rows, int columns, Toperation *M, std::string extraMessage);
    /**
         * @brief Metodo estatico que imprime las distintas matrices locales que tiene cada proceso con un retraso de su ID * 1000 ms. Pensada para ver los diferente bloques que pueda tener cada proceso
         * 
         * @param cpurank , Id de la cpu
         * @param cpuSize , Numero de procesadores que van a realizar la operacion matematica
         * @param rows , filas de la matriz
         * @param columns , columnas de la matriz
         * @param vector<Toperation*>M , vector que contiene las matrices de cada proceso que hay que mostrar
         * @param extraMessage , mensaje a mostrar por pantalla, puede ser vacio("") en caso de que no se quiera mostar nada
         */
    static void debugMatricesLocalDifferentCpus(int cpurank, int cpuSize, int rows, int columns, std::vector<Toperation *> M, std::string extraMessage);
    /**
         * @brief Metodo estatico que mira si dos matrices son iguales mediante la norma de Frobenius
         * 
         * @param A , Primera matriz a comparar
         * @param B , Segunda matriz a comparar
         * @param rows , Filas de las matrices
         * @param columns , Columnas de las matrices
         */
    static int checkEqualityOfMatrices(Toperation *A, Toperation *B, int rows, int columns);
    /**
         * @brief Metodo estatico que obtiene una matriz sin los 0s de las filas y columnas extendidas. DEPRECATED
         * 
         * @param rowsReal , Filas que deberia de tener la matriz sin ser extendida por 0s
         * @param columnsUsed , Columnas usadas en la matriz operacional
         * @param columnsReal , Columnas que deberia de tener la matriz sin ser extendida por 0s
         * @param matrix , matriz con los 0s
         * @return Toperation* , matriz sin los 0s
         */
    static Toperation *getMatrixWithoutZeros(int rowsReal, int columnsUsed, int columnsReal, Toperation *matrix);
    /**
         * @brief Metodo que comprueba si dos matrices se pueden multiplicar entre si
         * 
         * @param columnsA , columnas de la matriz A
         * @param rowsB , filas de la matriz B
         * @return true 
         * @return false 
         */
    static bool canMultiply(int columnsA, int rowsB);
    /**
         * @brief Metodo estatico que calcula el tamaño de la malla y de las matrices operacionales para realizar el calculo
         * 
         * @param rowsA , Filas de la matriz B
         * @param columnsA , columnas de la matriz A
         * @param rowsB , filas de la matriz B
         * @param columnsB , columnas de la matriz B
         * @param cpuSize numero de procesadores disponibles para realizar el calculo
         * @return OperationProperties 
         */
    static OperationProperties getMeshAndMatrixSize(int rowsA, int columnsA, int rowsB, int columnsB, int cpuSize);
    /**
         * @brief Metodo estatico que calcula la posicion del puntero unidimensional de un elemento de una matriz
         * 
         * @param columnSize , tamaño de las columnas de la matriz
         * @param rowIndex , fila del elemento al que se quiere acceder
         * @param columnIndex , columna del elemento al que se quiere acceder
         * @return int 
         */
    static int matrixCalculateIndex(int columnSize, int rowIndex, int columnIndex);
    /**
         * @brief Metodo estatico que reserva memoria para una matriz
         * 
         * @param rows , filas de la matriz
         * @param columns , columnas de la matriz
         * @return Toperation* , Puntero de la matriz que se ha reservado en memoria
         */
    static Toperation *matrixMemoryAllocation(int rows, int columns);
    /**
         * @brief Metodo estatico que se encarga de liberar la memoria del puntero de la Matriz que se pasa como argumento
         * 
         * @param matrix , Matriz que se va a liberar de la memoria
         */
    static void matrixFree(Toperation *matrix);
    /**
     * @brief 
     * 
     * @param isRandom , Indica si la matriz se va a generar de forma aleatorioa
     * @param fileName , Fichero del que se va a leer la matriz en el caso de que asi sea. Puede ser cualquier valor en caso de que no sea asi
     * @param rows , filas de la matriz
     * @param columns , columnas de la matriz
     * @param boundLower , limite inferior en el caso de que sea una matriz random. Puede ser cualquier valor en caso de que no sea asi
     * @param boundUpper , limite superior en el caso de que sea una matriz random. Puede ser cualquier valor en caso de que no sea asi
     * @return Toperation* 
     */
    static Toperation *ReadOrGenerateRandomMatrix(bool isRandom, const char *fileName, int &rows, int &columns, int boundLower, int boundUpper);
    /**
         * @brief Metodo estatico que multiplica 2 matrices A y B y suma el resultado en C mediante la libreria cblas Operacion C+=A*B
         * 
         * @param rowsA , Filas de A
         * @param columnsAorRowsB , Filas de B o columnas de A
         * @param columnsB , Columnas de B
         * @param A , Matriz A
         * @param B , Matriz B
         * @param C , Matriz a la cual es le va a sumar el resultado de A*B
         */
    static void matrixBlasMultiplication(int rowsA, int columnsAorRowsB, int columnsB, Toperation *A, Toperation *B, Toperation *C);
    /**
         * @brief Método estático que calcula la norma frobenius de una matriz
         * 
         * @param rows , Filas
         * @param columns , Columnas
         * @param A , Matriz 
         */
    static double frobeniusNormMatrixLapack(int rows, int columns, Toperation *A);
    /**
         * @brief Metodo estatico que realizada la multiplicacion entre dos matrices y suma el resultado obtenido en otra. Operacion C+=A*B. DEPRECATED. Se recomienda el uso de matrixBlasMultiplication
         * 
         * @param rowsA , Filas de A
         * @param columnsAorRowsB , Filas de B o columnas de A
         * @param columnsB , Columnas de B
         * @param A , Matriz A
         * @param B , Matriz B
         * @param C , Matriz a la cual es le va a sumar el resultado de A*B
         */
    static void Multiplicacion(int rowsA, int columnsAorRowsB, int columnsB, Toperation *A, Toperation *B, Toperation *C);

private:
    MatrixUtilities();
    /**
     * @brief Metodo privado que mira que calcula las propiedades de la operacion de forma individual para ese numero de cpus en cada lado de la malla
     * 
     * @tparam Toperation , Tipo basico de la operacion de la matriz
     * @param rowsA , filas de la matriz A
     * @param columnsAorRowsB , filas de B o columnas de A
     * @param columnsB , columnas de B
     * @param nCpusMesh1 , cpus en una dimension de la malla
     * @param nCpusMesh2 , cpus en otra dimension de la malla
     * @param isMeshRow ,si es verdadero la nCpusMesh1 va para meshRowSize, si es falso para meshColumnSize
     * @return OperationProperties , propiedades de la operacion
     */
    static OperationProperties calculateNonEqualMesh(int rowsA, int columnsAorRowsB, int columnsB, int nCpusMesh1, int nCpusMesh2, bool isMeshRow);
};
#endif