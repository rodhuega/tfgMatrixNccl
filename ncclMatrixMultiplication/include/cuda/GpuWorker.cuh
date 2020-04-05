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
        std::vector<cudaStream_t> streams;

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
         * @brief Agregas una matriz local a la gpu
         * 
         * @param gpumatrixLocal 
         */
        void setMatrixLocal(Toperation* gpumatrixLocal);
};