#pragma once

#include "vector"

#include "MatrixMain.cuh"
template <class Toperation>
class MatrixMain;


template <class Toperation>
class GpuWorker
{
    private:
        std::vector<Toperation*> gpuMatricesLocal;
        std::vector<cudaStream_t*> streams;

        int gpuRankWorld,gpuRankSystem;
        MatrixMain<Toperation>* matrixMainGlobal;
    public:
        GpuWorker(int gpuRankWorld,int gpuRankSystem,MatrixMain<Toperation>* matrixMainGlobal);
        int getGpuRankWorld();
        int getGpuRankSystem();
        /**
         * @brief Devuelve la matriz dentro de la gpu local solicitada
         * 
         * @param pos, posición de la matriz local en el vector
         * @return Toperation* 
         */
        Toperation* getMatrixLocal(int pos);
        /**
         * @brief Devuelve el vector que contiene todas las matrices locales de la gpu
         * 
         * @return vector<Toperation*> 
         */
        std::vector<Toperation*> getMatricesLocal();
        /**
         * @brief Devuelve el stream solicitado
         * 
         * @param pos, posición del stream en el vector
         * @return Toperation* 
         */
        cudaStream_t* getStream(int pos);
        /**
         * @brief Devuelve el vector que contiene todos los punteros a los streams
         * 
         * @return vector<Toperation*> 
         */
        std::vector<cudaStream_t*> getStreams();
        /**
         * @brief Agregas una matriz local a la gpu
         * 
         * @param gpumatrixLocal 
         */
        void setMatrixLocal(Toperation* gpumatrixLocal);
        /**
         * @brief Agregas una matriz local a la gpu
         * 
         * @param gpumatrixLocal 
         */
        void addStream(cudaStream_t* stream);
        /**
         * @brief Espera a todas las streams
         * 
         */
        void waitAllStreams();
};