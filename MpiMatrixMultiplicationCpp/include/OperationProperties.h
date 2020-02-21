#ifndef OperationProperties_H
#define OperationProperties_H

/**
 * @brief Struct que contiene las propiedades operacionales para realizar el calculo de la multiplicacion.
 * 
 */
struct OperationProperties
{
    /**
     * @brief Tamaño de las filas de la malla
     * 
     */
    int meshRowSize;
    /**
     * @brief Tamaño de las columnas de la malla
     * 
     */
    int meshColumnSize;
    /**
     * @brief Filas operacionales que tendra la matriz A
     * 
     */
    int rowsA;
    /**
     * @brief Columnas de la Matriz A o Filas de la Matriz B operacionales
     * 
     */
    int columnsAorRowsB;
    /**
     * @brief Columnas operacionales que tendra la matriz B
     * 
     */
    int columnsB;
    /**
     * @brief Numeros de 0s que tendra la matriz operacional al extenderse
     * 
     */
    int numberOf0;
    /**
     * @brief Numero de procesadores que realizaran la operacion de multiplicacion
     * 
     */
    int cpuSize;
    /**
     * @brief Numero de columnas que tendra la matriz A de forma local
     * 
     */
    int blockColumnSizeA;
    /**
     * @brief Numero de filas que tendra la matriz B de forma local
     * 
     */
    int blockRowSizeB;
    /**
     * @brief Incida si las propiedades antes indicadas son aptas para el calculo de la multiplicaicon de la matriz
     * 
     */
    bool candidate;
};

#endif