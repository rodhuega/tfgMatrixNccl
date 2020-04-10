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

/**
 * @brief Clase estática que contiene a las matrices y las características necesarias para realizar la múltiplicación de matrices mediante el algoritmo Summa
 * 
 * @tparam Toperation , tipo de la matriz(double,float) 
 */
template <class Toperation>
class NcclMultiplicationEnvironment
{
private:
    ncclComm_t *commOperation;
    ncclDataType_t basicOperationType;
    OperationType opType;
    bool printMatrix;
    int gpuSizeOperationWorld,gpuSizeOperationSystem,gpuSizeSystem,gpuSizeWorld,gpuRoot;
    std::unordered_map<std::string,MatrixMain<Toperation>*> matricesMatrixMain;
    std::vector<cudaStream_t*> cublasStreams;
    std::vector<cublasHandle_t*> cublasHandlers;

    /**
     * @brief Crea los comunicadores nccl correspondientes asignado sus propiedades a los elementos necesarios del vector commElements
     * 
     * @param commElements , vector que contiene cada uno de los elementos que tienen las caracteristicas de comunicación para la operación
     * @param dimensionLogicDevices , ids de los elementos lógicos que formaran parte del comunicador
     * @param setRowColor , si es true indica que el comunicador es para las filas, false es para las columnas
     */
    void createNcclCommunicator(std::vector<CommSummaElement*> &commElements,std::set<int> &dimensionLogicDevices,bool setRowColor);

    /** EN REALIDAD SOBRA LA MAYOR PARTE DE SU CÓDIGO INTERNO Y NO SE PORQUE NO ME LO PUEDO CARGAR. DA ERROR AL USAR si no setea commOperation waitAllCublasStreams()
     * @brief Método que crea el comunicador para solo las gpus que van a realizar la operacion multiplicativa.
     * 
     * @param gpuOperationSize , número de gpus que van a usarse en la operación
     */
    void setCommOperation(int gpuOperationSize);
    /**
     * @brief Método que realiza la multiplicacion de matrices de forma distribuida y devuelve la matriz local a cada gpu con el resultado. C=A*B
     * 
     * @param matrixLocalA , matriz parte izquierda
     * @param matrixLocalB , matriz parte derecha
     * @param meshRowsSize , tamaño de la malla de las filas
     * @param meshColumnsSize , tamaño de la malla de las columnas
     * @return MatrixMain<Toperation>* Matriz resultado 
     */
    MatrixMain<Toperation>* ncclSumma(MatrixMain<Toperation>* matrixLocalA, MatrixMain<Toperation>* matrixLocalB, int meshRowsSize, int meshColumnsSize);

public:
    /**
     * @brief Se construye el objeto que realizara las distitnas multiplicaciones en un entorno distribuido
     * 
     * @param gpuSizeWorld , Tamaño del número de gpus que se quiere que realicen las operaciones. (Puede ser mayor que el número físico, se usaran mas veces las gpus físicas)
     * @param gpuRoot , Id de la gpu que actura como root.
     * @param opType , Indica el tipo de operacion que se usara para las operaciones. Tiene que ser compatible con el tipo usado en la generacidad.MultDouble|MultFloat
     * @param printMatrix , indica si se van a imprimir las matrices idA e idB en la ejecución del programa
     */
    NcclMultiplicationEnvironment(int gpuSizeWorld,int gpuRoot,OperationType opType,bool printMatrix);
    /**
     * @brief Destructor del objeto. Libera los manejadores de cublas, sus streams y el comunicador.
     * 
     */
    ~NcclMultiplicationEnvironment();
    /**
     * @brief Añade o sustituye una nueva MatrixMain al entorno multiplicativo
     * 
     * @param id , identificador con el que se guardara la MatrixMain
     * @param matrixMainGlobal , matrixMain que se agregara al entorno
     */
    void setOrAddMatrixMain(std::string id,MatrixMain<Toperation> *matrixMainGlobal);
    /**
     * @brief Método que elimina una matriz del entorno multiplicativo. 
     * 
     * @param id , identificador de la matriz
     * @param freeMemory , true si se quiere liberar la memoria de esa matriz, false en caso contrario
     */
    void removeMatrixMain(std::string id,bool freeMemory);
    /**
     * @brief Método que devuelve un puntero a la MatrixMain solicitada
     * 
     * @param id , identificador de la MatrixMain que se desea recuperar
     * @return MatrixMain<Toperation>* 
     */
    MatrixMain<Toperation>* getMainMatrix(std::string id);
    /**
     * @brief Devuelve el tamaño de gpus lógicas de la operación
     * 
     * @return int 
     */
    int getGpuSizeOperationWorld();
    /**
     * @brief Devuelve el tamaño de gpus físicas de la operación
     * 
     * @return int 
     */
    int getGpuSizeOperationSystem();
    /**
     * @brief Devuelve el tamaño de gpus físicas del entorno
     * 
     * @return int 
     */
    int getGpuSizeSystem();
    /**
     * @brief Devuelve el tamaño de gpus lógicas del entorno
     * 
     * @return int 
     */
    int getGpuSizeWorld();
    /**
     * @brief Devuelve la id de la gpuRoot
     * 
     * @return int 
     */
    int getGpuRoot();
    void waitAllCublasStreams();
    /**
     * @brief Genera un identicador candidato para una matriz de forma aleatoria, este será el identificador en caso de que no exista ya una matriz con ese identificador.
     * 
     * @return std::string 
     */
    std::string generateRandomCandiateId();
    /**
     * @brief Genera un identicador para una matriz de forma aleatoria.
     * 
     * @return std::string 
     */
    std::string generateRandomId();
    /**
     * @brief Metodo que realizar la multiplicacion C=A*B
     * 
     * @param idA , id de la matriz A(Parte izquierda)
     * @param idB , id de la matriz B(Parte derecha)
     * @param idC , id de la matriz C(Resultado)
     */
     MatrixMain<Toperation> * performCalculations(std::string idA,std::string idB, std::string idC);

};