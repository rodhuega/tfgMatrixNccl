#pragma once

#include "cuda_runtime.h"
#include <cublas_v2.h>
#include "nccl.h"

#include <unordered_map>
#include <random>
#include <vector>
#include <set>
#include <iostream>
#include <tuple>
#include <string>

#include "OperationProperties.h"
#include "OperationType.h"
#include "MatrixUtilities.h"

#include "ErrorCheckingCuda.cuh"
#include "MatrixMain.cuh"
#include "CommSummaElement.cuh"

template <class Toperation>
class MatrixMain;

template <class Toperation>
class NcclMultiplicationEnvironment
{
private:
    ncclComm_t *commWorld,*commOperation;
    ncclDataType_t basicOperationType;
    OperationType opType;
    int gpuSizeOperationWorld,gpuSizeOperationSystem,gpuSizeSystem,gpuSizeWorld,gpuRoot;
    std::unordered_map<std::string,MatrixMain<Toperation>*> matricesMatrixMain;
    std::unordered_map<std::string,dimensions> matricesGlobalDimensions;
    std::vector<cudaStream_t*> cublasStreams;
    std::vector<cublasHandle_t*> cublasHandlers;

    void createNcclCommunicator(std::vector<CommSummaElement*> &commElements,std::set<int> &dimensionLogicDevices,bool setRowColor);

    /**
     * @brief Metodo que crea el comunicador para solo las gpus que van a realizar la operacion multiplicativa.
     * 
     * @param gpuOperationSize , número de gpus que van a usarse en la operación
     */
    void setCommOperation(int gpuOperationSize);
    //Modificar comenatarios
    /**
     * @brief Metodo que realiza la multiplicacion de matrices de forma distribuida y devuelve la matriz local a cada gpu con el resultado. C=A*B
     * 
     * @param matrixLocalA , matriz A parte local de la cpu 
     * @param matrixLocalB , matriz B parte local de la cpu 
     * @param meshRowsSize , tamaño de la malla de las filas
     * @param meshColumnsSize , tamaño de la malla de las columnas
     * @return MatrixMain<Toperation>* Matriz C resultado local del procesador
     */
    MatrixMain<Toperation>* mpiSumma(MatrixMain<Toperation>* matrixLocalA, MatrixMain<Toperation>* matrixLocalB, int meshRowsSize, int meshColumnsSize);

public:
    /**
     * @brief Se construye el objeto que realizara las distitnas multiplicaciones en un entorno distribuido
     * 
     * @param gpuSizeWorld , Tamaño del número de gpus que se quiere que realicen las operaciones. (Puede ser mayor que el número físico, se usaran mas veces las gpus físicas)
     * @param gpuRoot , Id de la gpu que actura como root.
     * @param opType , Indica el tipo de operacion que se usara para las operaciones. Tiene que ser compatible con el tipo usado en la generacidad
     */
    NcclMultiplicationEnvironment(int gpuSizeWorld,int gpuRoot,OperationType opType);
    /**
     * @brief Añade o sustituye una nueva MatrixMain al entorno multiplicativo
     * 
     * @param id , identificador con el que se guardara la MatrixMain
     * @param matrixMainGlobal , matrixMain que se agregara al entorno
     */
    void setOrAddMatrixMain(std::string id,MatrixMain<Toperation> *matrixMainGlobal);
    void removeMatrixMain(std::string id,bool freeMemory);
    /**
     * @brief Metodo que devuelve un puntero a la MatrixMain solicitada
     * 
     * @param id , identificador de la MatrixMain que se desea recuperar
     * @param create , booleano que indica si en caso de que no exista se debe de crear
     * @return MatrixMain<Toperation>* 
     */
    MatrixMain<Toperation>* getMainMatrix(std::string id,bool create);
    int getGpuSizeOperationWorld();
    int getGpuSizeOperationSystem();
    int getGpuSizeSystem();
    int getGpuSizeWorld();
    int getGpuRoot();
    void waitAllCublasStreams();
    std::string generateRandomCandiateId();
    std::string generateRandomId();
    std::vector<int> convertSetToVector(std::set<int> &s);
    /**
     * @brief Metodo que realizar la multiplicacion C=A*B
     * 
     * @param idA , id de la matriz A(Parte izquierda)
     * @param idB , id de la matriz B(Parte derecha)
     * @param idC , id de la matriz C(Resultado)
     * @param printMatrix 
     */
     MatrixMain<Toperation> * performCalculations(std::string idA,std::string idB, std::string idC,bool printMatrix);

};