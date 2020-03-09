#ifndef MpiMultiplicationEnvironment_H
#define MpiMultiplicationEnvironment_H

#include <unordered_map>
#include <tuple>
#include "MatrixUtilities.h"
#include "MatrixMain.h"
#include "MpiMatrix.h"
#include "OperationProperties.h"

/**
 * @brief Clase que realiza la operacion de multiplicacion distirbuida de las matrices.
 * 
 * @tparam Toperation , tipo de la matriz(double,float)
 */
template <class Toperation>
class MpiMultiplicationEnvironment
{
private:
    MPI_Datatype basicOperationType;
    MPI_Comm commOperation;
    int cpuRank,cpuOperationSize,cpuRoot,cpuSizeInitial;
    bool thisCpuPerformOperation;
    std::unordered_map<std::string,MatrixMain<Toperation>*> matricesMatrixMain;
    std::unordered_map<std::string,Toperation*> matricesGlobalSimplePointer;
    std::unordered_map<std::string,dimensions> matricesGlobalDimensions;
    
    /**
     * @brief Metodo que crea el comunicador para solo los procesos que van a realizar la operacion multiplicativa.
     * 
     * @param cpuOperationsSize 
     */
    void setCommOperation(int cpuOperationsSize);

public:
    /**
     * @brief Se construye el objeto que realizara la multiplicacion distribuida mediante el algoritmo suma
     * 
     * @param cpuRank , Id de la cpu
     * @param cpuRoot , Id de la cpu que actuara como root
     * @param cpuSizeInitial , Tamaño de la cpu del rango global de mpi(Numero de procesos disponibles para realizar la operacion)
     * @param basicOperationType Tipo de numero con el que se realizara la multiplicacion(Double, Int..)
     */
    MpiMultiplicationEnvironment(int cpuRank,int cpuRoot,int cpuSizeInitial,MPI_Datatype basicOperationType);
    /**
     * @brief Destructor del objeto
     * 
     */
    ~MpiMultiplicationEnvironment();
    /**
     * @brief Añade una nueva matriz al entorno multiplicativo con un puntero simple
     * 
     * @param id , identificador con el que se guardara la MatrixMain
     * @param matrixMainGlobal , matrixMain que se agregara al entorno
     * @param rows , filas de la matriz
     * @param columns , columnas de la matriz
     */
    void setOrAddMatrixGlobalSimplePointer(std::string id,Toperation *matrixMainGlobal,int rows,int columns);
    /**
     * @brief Metodo que devuelve un puntero a la matriz de puntero simple(La matriz esta no esta distirbuida)
     * 
     * @param id , identificador de la matriz que se desea recuperar
     * @return Toperation* 
     */
    Toperation* getMatrixGlobalSimplePointer(std::string id);
    /**
     * @brief Añade o sustituye una nueva MatrixMain al entorno multiplicativo
     * 
     * @param id , identificador con el que se guardara la MatrixMain
     * @param matrixMainGlobal , matrixMain que se agregara al entorno
     */
    void setOrAddMatrixMain(std::string id,MatrixMain<Toperation> *matrixMainGlobal);
    /**
     * @brief Metodo que devuelve un puntero a la MatrixMain solicitada
     * 
     * @param id , identificador de la MatrixMain que se desea recuperar
     * @param create , booleano que indica si en caso de que no exista se debe de crear
     * @return MatrixMain<Toperation>* 
     */
    MatrixMain<Toperation>* getMainMatrix(std::string id,bool create);
    /**
     * @brief Metodo que recupera una matriz distribuida
     * 
     * @param id 
     */
    void recoverDistributedMatrix(std::string id);
    /**
     * @brief Indica si esta cpu ha realizado la operacion
     * 
     * @return true 
     * @return false 
     */
    bool getIfThisCpuPerformOperation();
    /**
     * @brief Metodo que realiza todos los preparativos para realizar la multiplicacion C=A*B
     * 
     * @param idA , id de la matriz A(Parte izquierda)
     * @param idB , id de la matriz B(Parte derecha)
     * @param idC , id de la matriz C(Resultado)
     * @param printMatrix 
     */
    void PerformCalculations(std::string idA,std::string idB, std::string idC,bool printMatrix);
    ////////////////////////PASAR A PRIVATE//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief Metodo que realiza la multiplicacion de matrices de forma distribuida y devuelve la matriz local a cada cpu con el resultado. C=A*B
     * 
     * @param matrixLocalA , matriz A parte local de la cpu 
     * @param matrixLocalB , matriz B parte local de la cpu 
     * @param meshRowsSize , tamaño de la malla de las filas
     * @param meshColumnsSize , tamaño de la malla de las columnas
     * @return Toperation* Matriz C resultado local del procesador
     */
    Toperation* mpiSumma(MpiMatrix<Toperation> matrixLocalA, MpiMatrix<Toperation> matrixLocalB, int meshRowsSize, int meshColumnsSize);
};

#endif