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
 * @tparam Toperation , tipo de la matriz(double,int,float)
 */
template <class Toperation>
class MpiMultiplicationEnvironment
{
private:
    MPI_Datatype basicOperationType;
    MPI_Comm commOperation;
    int cpuRank,cpuSize,cpuRoot,cpuSizeInitial;
    std::unordered_map<std::string,MatrixMain<Toperation>*> matricesGlobalDistributed;
    std::unordered_map<std::string,MpiMatrix<Toperation>*> matricesLocalDistributed;
    std::unordered_map<std::string,Toperation*> matricesGlobalNonDistributed;
    std::unordered_map<std::string,dimensions> matricesGlobalDimensions;
    
    
    void setCommOperation(int cpuOperationsSize);

public:
    /**
     * @brief Se construye el objeto que realizara la multiplicacion distribuida mediante el algoritmo suma
     * 
     * @param cpuRank , Id de la cpu
     * @param cpuRoot , Id de la cpu que actuara como root
     * @param commOperation , Comunicador de los procesadores que realizaran el computo
     * @param basicOperationType Tipo de numero con el que se realizara la multiplicacion(Double, Int..)
     */
    MpiMultiplicationEnvironment(int cpuRank,int cpuRoot,int cpuSizeInitial,MPI_Datatype basicOperationType);
    /**
     * @brief Añade una nueva matriz distribuida al entorno multiplicativo de forma local para cada proceso
     * 
     * @param id , identificador con el que se guardara la MpiMatrix
     * @param mpiLocalMatrix* , MpiMatrix que se va a recuperar
     */
    void setNewMatrixLocalDistributed(std::string id,MpiMatrix<Toperation>* mpiLocalMatrix);
    /**
     * @brief Metodo que devuelve un puntero a la MpiMatrix Local de cada proceso que esta distribuida
     * 
     * @param id , identificador de la matriz que se desea recuperar
     * @return MpiMatrix<Toperation>*
     */
    MpiMatrix<Toperation>* getAMatrixLocalDistributed(std::string id);
    /**
     * @brief Añade una nueva matriz al entorno multiplicativo(La matriz esta no esta distirbuida)
     * 
     * @param id , identificador con el que se guardara la MatrixMain
     * @param matrixMainGlobal , matrixMain que se agregara al entorno
     * @param rows , filas de la matriz
     * @param columns , columnas de la matriz
     */
    void setNewMatrixGlobalNonDistributed(std::string id,Toperation *matrixMainGlobal,int rows,int columns);
    /**
     * @brief Metodo que devuelve un puntero a la MatrixMain solicitada(La matriz esta no esta distirbuida)
     * 
     * @param id , identificador de la matriz que se desea recuperar
     * @return Toperation* 
     */
    Toperation* getAMatrixGlobalNonDistributed(std::string id);
    /**
     * @brief Añade una nueva MatrixMain al entorno multiplicativo(La matriz esta distirbuida)
     * 
     * @param id , identificador con el que se guardara la MatrixMain
     * @param matrixMainGlobal , matrixMain que se agregara al entorno
     */
    void setNewMatrixGlobalDistributed(std::string id,MatrixMain<Toperation> *matrixMainGlobal);
    /**
     * @brief Metodo que devuelve un puntero a la MatrixMain solicitada(La matriz esta distirbuida)
     * 
     * @param id , identificador de la MatrixMain que se desea recuperar
     * @return MatrixMain<Toperation>* 
     */
    MatrixMain<Toperation>* getAMatrixGlobalDistributed(std::string id);
    /**
     * @brief Metodo que devuelve un puntero a la MatrixMain solicitada, si no existe el objeto lo crea aunque no lo distribuye, pero se prepara para ello
     * 
     * @param id , identificador de la matriz que se desea recuperar
     * @return Toperation* 
     */
    MatrixMain<Toperation>* getAMatrixGlobal(std::string id);

    void setNewMatrixLocalDistributedWithDimensions(std::string id, MpiMatrix<Toperation> *mpiLocalMatrix, int rows, int columns);

    void setAMatrixGlobalNonDistributedFromLocalDistributed(std::string id);
    /////////////////////////////////////////POR COMENTAR////////////////////////////////////////////////////////////////////////////////////////////////////////
    void PerformCalculations(std::string idA,std::string idB, std::string idC,bool printMatrix);
    ////////////////////////PASAR A PRIVATE//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief Metodo que realiza la multiplicacion de matrices de forma distribuida y devuelve la matriz local a cada cpu con el resultado. C=A*B
     * 
     * @param matrixLocalA , matriz A parte local de la cpu 
     * @param matrixLocalB , matriz B parte local de la cpu 
     * @param meshRowsSize , tamaño de la malla de las filas
     * @param meshColumnsSize , tamaño de la malla de las columnas
     * @return MpiMatrix<Toperation> Matriz C resultado local del procesador
     */
    MpiMatrix<Toperation> mpiSumma(MpiMatrix<Toperation> matrixLocalA, MpiMatrix<Toperation> matrixLocalB, int meshRowsSize, int meshColumnsSize);
};

#endif