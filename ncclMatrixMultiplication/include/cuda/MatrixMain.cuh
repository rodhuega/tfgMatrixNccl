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
        std::vector<int> blocksInitialPosition;

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
         * @brief Obtiene el valor de blockSize
         * 
         * @return int 
         */
        int getBlockSize();
        /**
         * @brief Obtiene el valor de blockRowSize
         * 
         * @return int 
         */
        int getBlockRowSize();
        /**
         * @brief Obtiene el valor de blockColumnSize
         * 
         * @return int 
         */
        int getBlockColumnSize();
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
         * @brief Obtiene todos los gpuWorkers de la matriz
         * 
         * @return std::vector<GpuWorker<Toperation>*> 
         */
        std::vector<GpuWorker<Toperation>*> getGpuWorkers();
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

        void setMatrixOperationProperties(int meshRowSize, int meshColumnSize, int blockRowSize, int blockColumnSize);
        int calculateRowColor(int gpuRank);
        int calculateColumnColor(int gpuRank);
        /**
         * @brief Devuelve la longitud de numero de elementos que hay que copiar
         * 
         * @param color , color del bloque de la matriz, Fila o columna a la que pertenece en la matriz global
         * @param meshDimensionSize , tamaño de la dimension de la malla
         * @param blockDimenensionSize , tamaño de la dimension elegida de ese bloque
         * @param dimensionUsed , tamaño de la dimension elegida en la matriz global con la que se opera(0s incluidos)
         * @param dimensionReal , tamaño real de la dimension elegida en la matriz global(0s no incluidos)
         * @return int 
         */
        int calculateBlockDimensionToCopy(int color, int meshDimensionSize, int blockDimenensionSize, int dimensionUsed, int dimensionReal);
        void waitAllStreamsOfAllWorkers();
        void distributeMatrixIntoGpus();

};