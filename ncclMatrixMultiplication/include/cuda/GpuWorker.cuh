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
        int gpuRankWorld,gpuRankSystem;
        MatrixMain<Toperation>* matrixMainGlobal;
    public:
        GpuWorker(int gpuRankWorld,int gpuRankSystem);
        int getGpuRankWorld();
        int getGpuRankSystem();
};