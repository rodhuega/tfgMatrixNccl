#pragma once

#include <vector>
#include <string>
#include <tuple>
#include <cmath>

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
         * @param deepCopy , indica si se van a hacer copias de sus punteros
         */
        void assignationToActualObject(const MatrixMain<Toperation>& B,bool deepCopy);
        /**
         * @brief Realización de la operación axpy entre dos matrices(suma), la matriz Y a la cual se le va a sumar X 
         * 
         * @param alpha , escalar que multiplicará a X
         * @param X , matriz que se sumará a X
         * @param Y , matriz Y la cual se sumará a X y almacenará el nuevo resultado
         */
        void axpy(const Toperation& alpha,MatrixMain<Toperation>& X,MatrixMain<Toperation>& Y);
    public:
        /**
         * @brief Constructor de MatrixMain. Crea una MatrixMain y la asigna a un NcclMultiplicationEnvironment
         * 
         * @param ncclMultEnv ,Entorno donde se usará esta matriz
         * @param rows , filas reales de la matriz
         * @param columns , columnas reales de la matriz
         */
        MatrixMain(NcclMultiplicationEnvironment<Toperation>* ncclMultEnv,int rows,int columns);
        /**
         * @brief Constructor de MatrixMain. Crea una MatrixMain y la asigna a un NcclMultiplicationEnvironment
         * 
         * @param ncclMultEnv ,Entorno donde se usará esta matriz
         * @param rows , filas reales de la matriz
         * @param columns , columnas reales de la matriz
         * @param *matrix , puntero de la matriz host
         */
        MatrixMain(NcclMultiplicationEnvironment<Toperation>* ncclMultEnv,int rows,int columns, Toperation* matrix);
        /**
         * @brief Constructor de MatrixMain a partir de otro. Copia profunda de la matriz del host si la hay y de sus gpuWorkers
         * 
         * @param maMain , MatrixMain del cual se va a copiar
         */
        MatrixMain(const MatrixMain<Toperation> &maMain);
        /**
         * @brief Constructor de MatrixMain de movimiento
         * 
         * @param B
         */
        MatrixMain(MatrixMain<Toperation> &&B);
        /**
         * @brief Destructor de MatrixMain que elimina todos los gpuWorkers asociados.
         * Si se ha activado antes el flag correspondiente a true mediante setDeleteMatrixHostAtDestroyment() también elimina el puntero de la matriz host en caso de que exista.
         * 
         */
        ~MatrixMain();
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
         * @brief Copia la matriz al puntero del host que se pasa como argumento
         * 
         * @param pointerMatrix , puntero donde sera copiada la matriz
         */
        void getHostMatrixInThisPointer(Toperation* pointerMatrix);
        /**
         * @brief Obtiene todos los gpuWorkers de la matriz
         * 
         * @return std::vector<GpuWorker<Toperation>*> 
         */
        std::vector<GpuWorker<Toperation>*> getGpuWorkers();
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
         * @brief Asigna si hay una matriz en el host. En caso de que se asigne que no y este la matriz, se liberan recursos y se asigna nullptr
         * 
         * @param isMatrixHostHere 
         */
        void setIsMatrixHostHere(bool isMatrixHostHere);
        /**
         * @brief Asigna una nueva matriz a matrixHost y destruye los gpuWorkers que existían antes y la matrixHost que hubiese antes
         * 
         * @param newMatrixHost 
         */
        void setMatrixHost(Toperation* newMatrixHost);
        /**
         * @brief Asigna un valor a todos los elementos de la matriz de matrixHost. Elimina la que había con anterioridad
         * 
         * @param valueForHost , valor de todos los elementos de la matriz
         */
        void setMatrixHostToFullValue(Toperation valueForHost);
        /**
         * @brief Asigna si se destruirá la matriz del host cuando se destruya el objeto
         * 
         * @param deleteMatrixHostAtDestroyment 
         */
        void setDeleteMatrixHostAtDestroyment(bool deleteMatrixHostAtDestroyment);
        /**
         * @brief Asigna si una matriz está distribuida o no. 
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
         * @brief Distribuye la matriz del host a la gpu como si se multiplicase consigo misma
         * 
         */
        void distributeMatrixMySelfIntoGpus();
        /**
         * @brief Devuelve la matriz de las gpus al host en el puntero indicado
         * 
         * @param pointerMatrix, puntero que contendrá la matriz
         */
        void recoverMatrixToHost(Toperation* pointerMatrix);
        /**
         * @brief Realización de la operación axpy entre dos matrices(suma), la matriz actual a la cual se le va a sumar X 
         * 
         * @param alpha , escalar que multiplicará a X
         * @param X , matriz que se sumará a X
         */
        void axpy(const Toperation& alpha,MatrixMain<Toperation>& X);
        /**
         * @brief Calcula y devuelve la norma 1 de una matriz. Máxima suma de sus columnas
         * 
         * @return Toperation 
         */
        Toperation norm1();
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
         * @brief Override del operador *= (dividir y asignación) en caso de un escalar
         * 
         * @param alpha , escalar por el que se dividirá la matriz
         * @return MatrixMain<Toperation>& 
         */
        MatrixMain<Toperation>& operator/=(const Toperation& alpha);
        /**
         * @brief Override del operador / (división) en caso de un escalar
         * 
         * @param alpha , escalar por el que se dividirá la matriz
         * @return MatrixMain<Toperation> 
         */
        MatrixMain<Toperation> operator/(const Toperation& alpha);
        /**
         * @brief Override de asignación (=) de MatrixMain
         * 
         * @param B , MatrixMain que contiene los valores a asignar.
         * @return MatrixMain<Toperation>& 
         */
        MatrixMain<Toperation>& operator=(const MatrixMain<Toperation>& B);
        /**
         * @brief Asignación de movimiento
         * 
         * @param B 
         * @return MatrixMain<Toperation>& 
         */
        MatrixMain<Toperation>& operator=(MatrixMain<Toperation>&& B);
        /**
         * @brief Override del operador +=(suma y asignación) de una matriz + la identidad multiplicada por una constante
         * 
         * @param constantAddition , constante que multiplicará a la identidad
         * @return MatrixMain<Toperation>& 
         */
        MatrixMain<Toperation>& operator+=(const Toperation& constantAddition);
        /**
         * @brief Override del operador + (suma) de una matriz + la identidad multiplicada por una constante
         * 
         * @param constantAddition , constante que multiplicará a la identidad
         * @return MatrixMain<Toperation> 
         */
        MatrixMain<Toperation> operator+(const Toperation& constantAddition);
        /**
         * @brief Override del operador -= (resta y asignación) de una matriz - la identidad multiplicada por una constante
         * 
         * @param constantSubstraction , constante que multiplicará a la identidad
         * @return MatrixMain<Toperation> 
         */
        MatrixMain<Toperation>& operator-=(const Toperation& constantSubstraction);
        /**
         * @brief Override del operador - (resta) de una matriz - la identidad multiplicada por una constante
         * 
         * @param constantSubstraction , constante que multiplicará a la identidad 
         * @return MatrixMain<Toperation> 
         */
        MatrixMain<Toperation> operator-(const Toperation& constantSubstraction);
        /**
         * @brief Override del operador +=(suma y asignación) de una matriz + otra matriz
         * 
         * @param maMain , matriz con la cual se sumará
         * @return MatrixMain<Toperation>& 
         */
        MatrixMain<Toperation>& operator+=(MatrixMain<Toperation>& maMain);
        /**
         * @brief Override del operador +(suma) de una matriz + otra matriz
         * 
         * @param maMain , matriz con la cual se sumará
         * @return MatrixMain<Toperation>
         */
        MatrixMain<Toperation> operator+(MatrixMain<Toperation>& maMain);
        /**
         * @brief Override del operador -=(resta y asignación) de una matriz - otra matriz
         * 
         * @param maMain , matriz con la cual se restará
         * @return MatrixMain<Toperation>& 
         */
        MatrixMain<Toperation>& operator-=(MatrixMain<Toperation>& maMain);
        /**
         * @brief Override del operador -(resta) de una matriz - otra matriz
         * 
         * @param maMain , matriz con la cual se restará
         * @return MatrixMain<Toperation>
         */
        MatrixMain<Toperation> operator-(MatrixMain<Toperation>& maMain);
        /**
         * @brief Override del operador del cambio de signo -. Cambia el signo de todos los elementos de una matriz
         * 
         * @return MatrixMain<Toperation> 
         */
        MatrixMain<Toperation> operator-();
        /**
         * @brief Override del operador + (suma) de la identidad multiplicada por un número  y otra matriz
         * 
         * @tparam To , tipo de la matriz
         * @param constantAddition , constante que multiplicará a la identidad
         * @param maMain , matriz a sumar
         * @return MatrixMain<To> 
         */
        template<typename To> friend MatrixMain<To> operator+(const double &constantAddition, const MatrixMain<To> &maMain);
        /**
         * @brief Override del operador * (multiplicación) en caso de un escalar. Escalar * matriz
         * 
         * @tparam To , tipo de la matriz
         * @param alpha , escalar por el que se dividirá la matriz
         * @param maMain , matriz que se va a multiplicar
         * @return MatrixMain<To> 
         */
        template<typename To> friend MatrixMain<To> operator*(const double &alpha, const MatrixMain<To> &maMain);
        /**
         * @brief Override del operador + (suma) de la identidad multiplicada por un número  y otra matriz
         * 
         * @tparam To , tipo de la matriz
         * @param constantSubstraction , constante que multiplicará a la identidad
         * @param maMain , matriz que va a ser restada
         * @return MatrixMain<To> 
         */
        template<typename To> friend MatrixMain<To> operator-(const double &constantSubstraction, const MatrixMain<To> &maMain);  
             

};