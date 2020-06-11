#pragma once

#include "cuda_runtime.h"
#include "cuda.h"
#include <cublas_v2.h>
#include "nccl.h"

#include <map>
#include <vector>
#include <set>
#include <iostream>
#include <tuple>
#include <string>

#include "OperationProperties.h"
#include "OperationType.h"

#include "MatrixUtilitiesCuda.cuh"
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
    ncclDataType_t basicOperationType;
    OperationType opType;
    bool printMatrix;
    int gpuSizeOperationWorld,gpuSizeOperationSystem,gpuSizeSystem,gpuSizeWorld,gpuRoot,lastMeshRowSize,lastMeshColumnSize,lastBlockRowSizeA,lastBlockColumnSizeA,lastBlockRowSizeB,lastBlockColumnSizeB;
    std::vector<cudaStream_t*> cublasStreams;
    std::vector<cublasHandle_t*> cublasHandlers;
    //La clave es el tamaño del meshRowSize. El valor es una clave con la siguiente estructura:
    //Elemento 0, vector que tiene los comunicadores y sus propiedades.
    //Elemento 1. vector con los colores lógicos de las filas
    //Elemento 2. vector con los colores lógicos de las columnas
    std::map<int, std::tuple<std::vector<CommSummaElement*>,std::vector<std::set<int>>,std::vector<std::set<int>>>> summaComms;
    //Matrices buffer
    std::vector<Toperation*> gpuAuxiliarMatricesA,gpuAuxiliarMatricesB;


    /**
     * @brief Crea los comunicadores nccl correspondientes asignado sus propiedades a los elementos necesarios del vector commElements
     * 
     * @param commElements , vector que contiene cada uno de los elementos que tienen las caracteristicas de comunicación para la operación
     * @param dimensionLogicDevices , ids de los elementos lógicos que formaran parte del comunicador
     * @param setRowColor , si es true indica que el comunicador es para las filas, false es para las columnas
     */
    void createNcclCommunicator(std::vector<CommSummaElement*> &commElements,std::set<int> &dimensionLogicDevices,bool setRowColor);
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
    /**
     * @brief Método que elimina las matrices usadas como buffer
     * 
     */
    void eraseBufferMatrix();

public:
    /**
     * @brief Se construye el objeto que realizara las distitnas multiplicaciones en un entorno distribuido
     * 
     * @param gpuSizeWorld , Tamaño del número de gpus que se quiere que realicen las operaciones. (Puede ser mayor que el número físico, se usarán mas veces las gpus físicas)
     * @param gpuRoot , Id de la gpu que actuará como root.
     * @param opType , Indica el tipo de operacion que se usará para las operaciones. Tiene que ser compatible con el tipo usado en la generacidad.MultDouble|MultFloat
     * @param printMatrix , indica si se van a imprimir las matrices idA e idB en la ejecución del programa
     */
    NcclMultiplicationEnvironment(int gpuSizeWorld,int gpuRoot,OperationType opType,bool printMatrix);
    /**
     * @brief Destructor del objeto. Libera los manejadores de cublas, sus streams y el comunicador.
     * 
     */
    ~NcclMultiplicationEnvironment();
    /**
     * @brief Devuelve el tipo de nccl que se está usando
     * 
     * @return ncclDataType_t 
     */
    ncclDataType_t getBasicOperationType();
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
    /**
     * @brief Cambia el valor de gpuSizeOperationWorld
     * 
     * @param gpuSizeOperationWorld , nuevo valor de gpuSizeOperationWorld
     */
    void setGpuSizeOperationWorld(int gpuSizeOperationWorld);
    /**
     * @brief Cambia el valor de gpuSizeOperationSystem
     * 
     * @param gpuSizeOperationSystem , nuevo valor de gpuSizeOperationSystem
     */
    void setGpuSizeOperationSystem(int gpuSizeOperationSystem);
    /**
     * @brief Método que devuelve el tipo de las operaciones que se están llevan a cabo
     * 
     * @return OperationType MultDouble|MultFloat
     */
    OperationType getOperationType();
    /**
     * @brief Método que devuelve todos los cublas handlers del entorno
     * 
     * @return std::vector<cublasHandle_t*> 
     */
    std::vector<cublasHandle_t*> getCublasHandlers();
    /**
     * @brief Método que espera a que se completen todas las operaciones de cublas que hay en curso
     * 
     */
    void waitAllCublasStreams();
    /**
     * @brief Recupera o crear y recupera los comunicadores para el algoritmo summa o el cálculo de la norma de una matriz
     * 
     * @param meshRowSize , tamaño de la malla por filas
     * @param meshColumnSize , tamaño de la malla por columnas
     * @param matrixA , MatrixMain sobre la que se crearán los comunicadores en caso de que haga falta
     * @return std::tuple<std::vector<CommSummaElement*>,std::vector<std::set<int>>,std::vector<std::set<int>>> , Elemento 0, vector que tiene los comunicadores y sus propiedades. Elemento 1. vector con los colores lógicos de las filas, Elemento 2. vector con los colores lógicos de las columnas
     */
    std::tuple<std::vector<CommSummaElement*>,std::vector<std::set<int>>,std::vector<std::set<int>>> getOrCreateCommunicators(int meshRowSize, int meshColumnSize,MatrixMain<Toperation>* matrixA);
    /**
     * @brief Método que realizar la multiplicacion C=A*B
     * 
     * @param idA , id de la matriz A(Parte izquierda)
     * @param idB , id de la matriz B(Parte derecha)
     * @param idC , id de la matriz C(Resultado)
     */
     MatrixMain<Toperation>& performCalculations(MatrixMain<Toperation>& ma,MatrixMain<Toperation>& mb);

};