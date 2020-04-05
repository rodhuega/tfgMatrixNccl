#pragma once

#include <vector>
#include <string>

#include "GpuWorker.cuh"
#include "MatrixUtilitiesCuda.cuh"
#include "NcclMultiplicationEnvironment.cuh"

template <class Toperation>
class GpuWorker;

template <class Toperation>
class NcclMultiplicationEnvironment;

template <class Toperation>
class MatrixMain
{
    private:
        std::string id;
        Toperation* hostMatrix;
        std::vector<GpuWorker<Toperation>*> gpuWorkers;
        NcclMultiplicationEnvironment<Toperation>* ncclMultEnv;
        int rowsReal;
        int rowsUsed;
        int columnsReal;
        int columnsUsed;
        bool isDistributed;
        bool isMatrixHostHere;
        int blockRowSize,blockColumnSize,blockSize,meshRowSize,meshColumnSize,numberOfRowBlocks,numberOfColumnBlocks,numberOfTotalBlocks;

    public:
        /**
         * @brief Construct a new Matrix Main object
         * 
         * @param ncclMultEnv 
         */
        MatrixMain(NcclMultiplicationEnvironment<Toperation>* ncclMultEnv, std::string id,int rows,int columns);
        MatrixMain(NcclMultiplicationEnvironment<Toperation>* ncclMultEnv,std::string id,int rows,int columns, Toperation* matrix);
        /**
         * @brief Indica si una matriz esta distribuida o no.
         * 
         * @return true 
         * @return false 
         */
        bool getIsDistributed();
        /**
         * @brief Obtiene el valor de filas reales de la verdadera matriz
         * 
         * @return int 
         */
        int getRowsReal();
        /**
         * @brief Obtiene el valor de filas que se usara para operar, en caso de con coincidir con columnsReal significa que el exceso son 0
         * 
         * @return int 
         */
        int getRowsUsed();
        /**
         * @brief Obtiene el valor de columnas reales de la verdadera matriz
         * 
         * @return int 
         */
        int getColumnsReal();
        /**
         * @brief Obtiene el valor de columnas que se usara para operar, en caso de con coincidir con columnsReal significa que el exceso son 0
         * 
         * @return int 
         */
        int getColumnsUsed();
        /**
         * @brief Indica si hay una matriz global
         * 
         * @return int 
         */
        bool getIsMatrixHostHere();
        /**
         * @brief Obtiene el puntero de la matriz global del host.
         * 
         * @return Toperation* 
         */
        Toperation *getHostMatrix();
        /**
         * @brief Asigna el valor de filas que se usara para operar, en caso de con coincidir con columnsReal significa que el exceso son 0
         * 
         * @param rowsUsed 
         */
        void setRowsUsed(int rowsUsed);
        /**
         * @brief Asigna el valor de columnas que se usara para operar, en caso de con coincidir con columnsReal significa que el exceso son 0
         * 
         * @param columnsUsed 
         */
        void setColumnsUsed(int columnsUsed);
        /**
         * @brief Asigna si hay una matriz en el host
         * 
         * @param isMatrixHostHere 
         */
        void setIsMatrixHostHere(bool isMatrixHostHere);
        /**
         * @brief Asigna si una matriz esta distribuida o no
         * 
         * @param isDistributed 
         */
        void setIsDistributed(bool isDistributed);

        void setBlockAndMeshSize(int meshRowSize, int meshColumnSize, int blockRowSize, int blockColumnSize);
        void distributeMatrixIntoGpus();

};