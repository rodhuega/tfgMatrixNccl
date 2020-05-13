#pragma once

#include <vector>

#include "MatrixMain.cuh"
#include "MatrixUtilitiesCuda.cuh"
template <class Toperation>
class MatrixMain;

/**
 * @brief Clase que actua de gpu lógica en una matriz y contiene las distintas matrices de una gpu.
 * 
 * @tparam Toperation , tipo de la matriz(double,float)  
 */
template <class Toperation>
class GpuWorker
{
    private:
        std::vector<Toperation*> gpuMatricesLocal;
        std::vector<cudaStream_t*> streams;

        int gpuRankWorld,gpuRankSystem;
        MatrixMain<Toperation>* matrixMainGlobal;
    public:
        /**
         * @brief Constructor de un GpuWorker que actua como gpu lógica para esa matriz
         * 
         * @param gpuRankWorld , rango lógico del GpuWorker
         * @param gpuRankSystem , rango físico del GpuWorker
         * @param matrixMainGlobal , punterro al objeto matriz que contiene el resto de propiedades de la matriz.
         */
        GpuWorker(int gpuRankWorld,int gpuRankSystem,MatrixMain<Toperation>* matrixMainGlobal);
        /**
         * @brief Constructor de GpuWorker a partir de otro. Genera nuevos streams y copia profunda de sus matrices
         * 
         * @param gpuW , gpuWorker del cual se va a copiar
         */
        GpuWorker(const GpuWorker<Toperation> &gpuW);
        /**
         * @brief Constructor de GpuWorker de movimiento
         * 
         * @param gpuW
         */
        GpuWorker(GpuWorker<Toperation> &&gpuW);
        /**
         * @brief Destructor de GpuWorker. Libera las matrices de la gpu y destruye sus streams
         * 
         */
        ~GpuWorker();
        /**
         * @brief Asignación de movimiento
         * 
         * @param gpuW 
         * @return GpuWorker& 
         */
        GpuWorker& operator=(GpuWorker&& gpuW);
        /**
         * @brief Devuelve el rango lógico del GpuWorker
         * 
         * @return int 
         */
        int getGpuRankWorld();
        /**
         * @brief Devuelve el rango físico del GpuWorker
         * 
         * @return int 
         */
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
         * @brief Devuelve el vector que contiene todos los punteros de los streams
         * 
         * @return vector<Toperation*> 
         */
        std::vector<cudaStream_t*> getStreams();
        /**
         * @brief Agregas una matriz local a la gpu
         * 
         * @param gpumatrixLocal 
         */
        void addMatrixLocal(Toperation* gpumatrixLocal);
        /**
         * @brief Agregas un stream a la gpu
         * 
         * @param stream 
         */
        void addStream(cudaStream_t* stream);
        /**
         * @brief Espera a todas los streams del GpuWorker
         * 
         */
        void waitAllStreams();
};