#pragma once

#include <vector>
#include <string>
#include <tuple>

#include <MatrixUtilities.h>

#include "GpuWorker.cuh"
#include "MatrixUtilitiesCuda.cuh"
#include "NcclMultiplicationEnvironment.cuh"

template <class Toperation>
class GpuWorker;

template <class Toperation>
class NcclMultiplicationEnvironment;

/**
 * @brief Clase que contiene todas las propiedades de una matriz asi como los GpuWorkers asociados a ella
 * 
 * @tparam Toperation , tipo de la matriz(double,float)  
 */
template <class Toperation>
class MatrixMain
{
    private:
        std::string id;
        Toperation* hostMatrix;
        std::vector<GpuWorker<Toperation>*> gpuWorkers;
        std::vector<int> blocksInitialPosition;
        //Valores de la tuppla
        //1º Índice de la matriz global del primer elemento de la diagonal del bloque. -1 en caso de que no tenga
        //2º Índice local del bloque del primer elemento de la diagonal
        //3º Longitud de la diagonal en el bloque
        std::vector<std::tuple<int,int,int>>blocksInitialPositionDiagonal;//Falta copia y destructor

        NcclMultiplicationEnvironment<Toperation>* ncclMultEnv;
        int rowsReal;
        int rowsUsed;
        int columnsReal;
        int columnsUsed;
        Toperation alphaGemm;
        bool isDistributed;
        bool isMatrixHostHere;
        bool deleteMatrixHostAtDestroyment;
        bool deleteObjectAtDestroyment;
        int blockRowSize,blockColumnSize,blockSize,meshRowSize,meshColumnSize,numberOfRowBlocks,numberOfColumnBlocks,numberOfTotalBlocks;

        /**
         * @brief Metodo que elimina los gpuWorkers del objeto
         * 
         */
        void deleteGpuWorkers();
        /**
         * @brief Metodo que asigna al objeto actual otro objeto
         * 
         * @param B , objeto que contiene las nuevas propiedades
         * @param sameId , indica si va a tener la misma id
         * @param deepCopy , indica si se van a hacer copias de sus punteros
         */
        void assignationToActualObject(const MatrixMain<Toperation>& B,bool sameId,bool deepCopy);
    public:
        /**
         * @brief Constructor de MatrixMain. Crea una MatrixMain y la asigna a un NcclMultiplicationEnvironment
         * 
         * @param ncclMultEnv ,Entorno donde se usará esta matriz
         * @param id , identificador de la matriz. Si se pasa "" se generara uno de forma automática
         * @param rows , filas reales de la matriz
         * @param columns , columnas reales de la matriz
         */
        MatrixMain(NcclMultiplicationEnvironment<Toperation>* ncclMultEnv, std::string id,int rows,int columns);
        /**
         * @brief Constructor de MatrixMain. Crea una MatrixMain y la asigna a un NcclMultiplicationEnvironment
         * 
         * @param ncclMultEnv ,Entorno donde se usará esta matriz
         * @param id , identificador de la matriz. Si se pasa "" se generara uno de forma automática
         * @param rows , filas reales de la matriz
         * @param columns , columnas reales de la matriz
         * @param *matrix , puntero de la matriz host
         */
        MatrixMain(NcclMultiplicationEnvironment<Toperation>* ncclMultEnv,std::string id,int rows,int columns, Toperation* matrix);
        /**
         * @brief Constructor de MatrixMain a partir de otro. Copia profunda de la matriz del host si la hay y de sus gpuWorkers
         * 
         * @param maMain , MatrixMain del cual se va a copiar
         */
        MatrixMain(const MatrixMain<Toperation> &maMain);
        /**
         * @brief Destructor de MatrixMain que elimina todos los gpuWorkers asociados.
         * Si se ha activado antes el flag correspondiente a true mediante setDeleteMatrixHostAtDestroyment() tambien elimina el puntero de la matriz host en caso de que exista.
         * 
         */
        ~MatrixMain();
        /**
         * @brief Obtiene la id de la matriz
         * 
         * @return std::string 
         */
        std::string  getId();
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
         * @brief Obtiene el valor de filas que se usará para operar, en caso de no con coincidir con columnsReal significa que el exceso son 0
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
         * @brief Obtiene el valor de columnas que se usará para operar, en caso de no con coincidir con columnsReal significa que el exceso son 0
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
         * @brief Obtiene el tamaño de la malla de las columnas
         * 
         * @return int 
         */
        int getMeshColumnSize();
        /**
         * @brief Obtiene el tamaño de la malla de las filas
         * 
         * @return int 
         */
        int getMeshRowSize();
        /**
         * @brief Obtiene el valor del escalar alfa en la operacion GEMM
         * 
         * @return Toperation 
         */
        Toperation getAlphaGemm();
        /**
         * @brief Indica si hay una matriz global
         * 
         * @return int 
         */
        bool getIsMatrixHostHere();
        /**
         * @brief Indica si se destruirá la matriz del host cuando se elimine el objeto
         * 
         * @return int 
         */
        bool getDeleteMatrixHostAtDestroyment();
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
         * @brief Cambia la id de la matriz.
         * 
         * @param id 
         */
        void setId(std::string id);
        /**
         * @brief Asigna el valor de filas que se usará para operar, en caso de no con coincidir con columnsReal significa que el exceso son 0
         * 
         * @param rowsUsed 
         */
        void setRowsUsed(int rowsUsed);
        /**
         * @brief Asigna el valor de columnas que se usará para operar, en caso de no con coincidir con columnsReal significa que el exceso son 0
         * 
         * @param columnsUsed 
         */
        void setColumnsUsed(int columnsUsed);
        /**
         * @brief Asigna el valor del escalar alfa para la operacion GEMM
         * 
         * @param alphaGemm 
         */
        void setAlphaGemm(Toperation alphaGemm);
        /**
         * @brief Asigna si hay una matriz en el host,En caso de que se asigne que no y este la matriz, se liberan recursos y se asgina nullptr
         * 
         * @param isMatrixHostHere 
         */
        void setIsMatrixHostHere(bool isMatrixHostHere);
        /**
         * @brief Asigna si se destruirá la matriz del host cuando se destruya el objeto
         * 
         * @param deleteMatrixHostAtDestroyment 
         */
        void setDeleteMatrixHostAtDestroyment(bool deleteMatrixHostAtDestroyment);
        /**
         * @brief Asigna si una matriz esta distribuida o no. 
         * 
         * @param isDistributed 
         */
        void setIsDistributed(bool isDistributed);
        /**
         * @brief Asigna las propiedades de la matriz para la operacion que se va a realizar
         * 
         * @param meshRowSize , tamaño de la malla en número de filas
         * @param meshColumnSize , tamaño de la malla en número de columnas
         * @param blockRowSize , tamaño de las filas en un bloque
         * @param blockColumnSize , tamaño de las columnas en un bloque
         */
        void setMatrixOperationProperties(int meshRowSize, int meshColumnSize, int blockRowSize, int blockColumnSize);
        /**
         * @brief Calcula el color de la fila respecto el rango de la gpu dado en la matriz
         * 
         * @param gpuRank , rango de la gpu
         * @return int 
         */
        int calculateRowColor(int gpuRank);
        /**
         * @brief Calcula el color de la columna respecto el rango de la gpu dado en la matriz
         * 
         * @param gpuRank , rango de la gpu
         * @return int 
         */
        int calculateColumnColor(int gpuRank);
        /**
         * @brief Devuelve la longitud de número de elementos que hay que copiar
         * 
         * @param color , color del bloque de la matriz, Fila o columna a la que pertenece en la matriz global
         * @param meshDimensionSize , tamaño de la dimensión de la malla
         * @param blockDimenensionSize , tamaño de la dimensión elegida de ese bloque
         * @param dimensionUsed , tamaño de la dimensión elegida en la matriz global con la que se opera(0s incluidos)
         * @param dimensionReal , tamaño real de la dimensión elegida en la matriz global(0s no incluidos)
         * @return int 
         */
        int calculateBlockDimensionToCopy(int color, int meshDimensionSize, int blockDimenensionSize, int dimensionUsed, int dimensionReal);
        /**
         * @brief Espera a que acaben todos los streams de los GpuWorkers(gpus lógicas)
         * 
         */
        void waitAllStreamsOfAllWorkers();
        /**
         * @brief Distribuye la matriz del host a las gpus
         * 
         */
        void distributeMatrixIntoGpus();
        /**
         * @brief Devuelve la matriz de las gpus al host
         * 
         */
        void recoverMatrixToHost();
        /**
         * @brief Override del operador *= (multiplicación y asignación) en caso entre matrices
         * 
         * @param B , La otra matriz por la cual se multiplica
         * @return MatrixMain<Toperation> 
         */
        MatrixMain<Toperation>& operator*=(MatrixMain<Toperation>& B );
        /**
         * @brief Override del operador * (multiplicación) en caso entre matrices
         * 
         * @param B , La otra matriz por la cual se multiplica
         * @return MatrixMain<Toperation> 
         */
        MatrixMain<Toperation>& operator*(MatrixMain<Toperation>& B);
        /**
         * @brief Override del operador *= (multiplicación y asignación) en caso de un escalar
         * 
         * @param alpha , escalar por el que se multiplicará la matriz
         * @return MatrixMain<Toperation>& 
         */
        MatrixMain<Toperation>& operator*=(const Toperation& alpha);
        /**
         * @brief Override del operador * (multiplicación) en caso de un escalar
         * 
         * @param alpha , escalar por el que se multiplicará la matriz
         * @return MatrixMain<Toperation> 
         */
        MatrixMain<Toperation> operator*(const Toperation& alpha);
        /**
         * @brief Override de asignación (=) de MatrixMain
         * 
         * @param B , MatrixMain que contiene los valores a asignar.
         * @return MatrixMain<Toperation>& 
         */
        MatrixMain<Toperation>& operator=(const MatrixMain<Toperation>& B);
        //W.I.P
        MatrixMain<Toperation>& operator+=(const Toperation& constantAddition);
        //W.I.P
        MatrixMain<Toperation>& operator-=(const Toperation& constantAddition);


};