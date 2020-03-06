#ifndef MpiMatrix_H
#define MpiMatrix_H

#include "mpi.h"
#include "vector"
#include <unistd.h>
#include <cblas.h>
#include "MatrixUtilities.h"

/**
 * @brief Clase que contiene informacion local para la multiplicacion de matrices
 * 
 * @tparam Toperation , tipo de la matriz(double,int,float)
 */
template <class Toperation>
class MpiMatrix
{
private:
    MPI_Comm commOperation;
    MPI_Datatype basicOperationType,matrixLocalType;
    Toperation* matrixLocal;
    int rowSize,columnSize,blockRowSize,blockColumnSize,blockSize,cpuRank,cpuSize,meshRowSize,meshColumnSize;
    std::vector<int> sendCounts;
    std::vector<int> blocks;

public:
    /**
     * @brief Constructor del objeto MpiMatrix que contiene infomacion local para la multiplicacion de matrices distribuida
     * 
     * @param cpuSize , numero de cpus que van a realizar el calculo
     * @param cpuRank , id del procesador
     * @param meshRowSize , tamaño de las filas de la malla
     * @param meshColumnSize , tamaño de las columnas de la malla
     * @param rowSize , numero de filas de la matriz global
     * @param columnSize , numero de columnas de la matriz global
     * @param commOperation , comunicador de los procesadores que van a realizar el calculo
     * @param basicOperationType tipo de mpi con el que se va a realizar la operacion (MPI_DOUBLE,MPI_INT)
     */
    MpiMatrix(int cpuSize,int cpuRank,int meshRowSize,int meshColumnSize,int rowSize,int columnSize,MPI_Comm commOperation,MPI_Datatype basicOperationType);
    /**
     * @brief Destructor del objeto
     * 
     */
    ~MpiMatrix();
    /**
     * @brief Devuelve el tamaño del las filas del bloque
     * 
     * @return int 
     */
    int getBlockRowSize();
    /**
     * @brief Devuelve el tamaño de las columnas del bloque
     * 
     * @return int 
     */
    int getBlockColumnSize();
    /**
     * @brief Devuelve el tamaño de las filas de la matriz operacional global
     * 
     * @return int 
     */
    int getRowSize();
    /**
     * @brief Devuelve el tamaño de las columnas de la matriz operacional global
     * 
     * @return int 
     */
    int getColumnSize();
    /**
     * @brief Devuelve el tamaño de las filas de la malla
     * 
     * @return int 
     */
    int getMeshRowSize();
    /**
     * @brief Devuelve el tamaño de las columnas de la malla
     * 
     * @return int 
     */
    int getMeshColumnSize();
    /**
     * @brief Devuelve el tamaño del bloque(numero de elementos locales de la matriz)
     * 
     * @return int 
     */
    int getBlockSize();
    /**
     * @brief Asignas la matriz local
     * 
     * @param matrixLocal 
     */
    void setMatrixLocal(Toperation* matrixLocal);
    /**
     * @brief Devuelve la matriz local
     * 
     * @return Toperation* 
     */
    Toperation* getMatrixLocal();
    /**
     * @brief Distribuye una matriz global entre distintos procesos mediante send y recv, 
     * al final de este metodo queda asignada al objeto MpiMatrix una matriz local para cada proceso
     * 
     * @param matrixGlobal , matriz global a distribuir
     * @param root , proceso que contiene la matriz global y procede a distribuirlo
     */
    void mpiDistributeMatrixSendRecv(Toperation *matrixGlobal,int root);
    /**
     * @brief Distribuye una matriz global entre distintos procesos, 
     * al final de este metodo queda asignada al objeto MpiMatrix una matriz local para cada proceso
     * 
     * @param matrixGlobal , matriz global a distribuir
     * @param root , proceso que contiene la matriz global y procede a distribuirlo
     */
    void mpiDistributeMatrix(Toperation *matrixGlobal,int root);
    /**
     * @brief Recupera una matriz distribuida mediante gatherV
     * 
     * @param root , proceso donde se quiere recuperar la matrix
     * @return Toperation* , matriz global recuperada
     */
    Toperation *mpiRecoverDistributedMatrixGatherV(int root);
    /**
     * @brief Recupera una matriz distribuida mediante Reduce
     * 
     * @param root , proceso donde se quiere recuperar la matrix
     * @return Toperation* , matriz global recuperada
     */
    Toperation* mpiRecoverDistributedMatrixReduce(int root);
};
#endif