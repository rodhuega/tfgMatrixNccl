#pragma once

/**
 * @brief Struct que contiene las propiedades operacionales para realizar el cálculo de la multiplicación.
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
     * @brief Filas operacionales que tendrá la matriz A
     * 
     */
    int rowsA;
    /**
     * @brief Columnas de la Matriz A o Filas de la Matriz B operacionales
     * 
     */
    int columnsAorRowsB;
    /**
     * @brief Columnas operacionales que tendrá la matriz B
     * 
     */
    int columnsB;
    /**
     * @brief Números de 0s que tendrá la matriz operacional al extenderse
     * 
     */
    int numberOf0;
    /**
     * @brief Número de gpus que realizarán la operacion de multiplicacion
     * 
     */
    int gpuSize;
    /**
     * @brief Número de filas que tendra la matriz A de forma local
     * 
     */
    int blockRowSizeA;
    /**
     * @brief Número de columnas que tendra la matriz A de forma local
     * 
     */
    int blockColumnSizeA;
    /**
     * @brief Número de filas que tendra la matriz B de forma local
     * 
     */
    int blockRowSizeB;
    /**
     * @brief Número de columnas que tendra la matriz B de forma local
     * 
     */
    int blockColumnSizeB;
    /**
     * @brief Indica si las propiedades antes indicadas son aptas para el cálculo de la multiplicación de la matriz
     * 
     */
    bool candidate;
};