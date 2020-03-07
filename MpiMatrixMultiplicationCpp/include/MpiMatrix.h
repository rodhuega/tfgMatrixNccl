#ifndef MpiMatrix_H
#define MpiMatrix_H

#include "mpi.h"
#include "vector"
#include <unistd.h>
#include <cblas.h>
#include "MatrixUtilities.h"
#include "MatrixMain.h"

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
    std::vector<Toperation*> matricesLocal;
    int blockRowSize,blockColumnSize,blockSize,cpuRank,cpuSize,meshRowSize,meshColumnSize,rowColor,columnColor,numberOfRowBlocks,numberOfColumnBlocks,numberOfTotalBlocks;
    std::vector<int> sendCounts;
    std::vector<int> blocks;
    MatrixMain<Toperation>* matrixMainGlobal;

    int calculateRowColor(int cpuRank);
    int calculateColumnColor(int cpuRank);

public:
    /**
     * @brief Constructor del objeto MpiMatrix que contiene infomacion local para la multiplicacion de matrices distribuida
     * 
     * @param cpuSize , numero de cpus que van a realizar el calculo
     * @param cpuRank , id del procesador
     * @param meshRowSize , tamaño de las filas de la malla
     * @param meshColumnSize , tamaño de las columnas de la malla
     * @param matrixGlobal , Objeto que contiene las propiedades de la matriz global
     * @param commOperation , comunicador de los procesadores que van a realizar el calculo
     * @param basicOperationType tipo de mpi con el que se va a realizar la operacion (MPI_DOUBLE,MPI_INT)
     */
    MpiMatrix(int cpuSize,int cpuRank,int meshRowSize,int meshColumnSize,int blockRowSize, int blockColumnSize,MatrixMain<Toperation>* mm,MPI_Comm commOperation,MPI_Datatype basicOperationType);
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
     * @brief Devuelve el vector que contiene todas las matrices locales
     * 
     * @return vector<Toperation*> 
     */
    std::vector<Toperation*> getMatricesLocal();
    /**
     * @brief Devuelve la matriz local solicitada
     * 
     * @param posicion de la matriz local en el vector
     * @return Toperation* 
     */
    Toperation* getMatrixLocal(int pos);
    /**
     * @brief Metodo que devuelve un puntero al objeto MatrixMain que contiene las propiedades de la matriz global
     * 
     * @return MatrixMain<Toperation>* 
     */
    MatrixMain<Toperation>* getMatrixMain();
    int calculateBlockDimensionSizeSend(int color, int meshDimensionSize, int blockDimenensionSize, int dimensionUsed, int dimensionReal);
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
    /**
     * @brief Recupera una matriz distribuida mediante Send Recv
     * 
     * @param root , proceso donde se quiere recuperar la matrix
     * @return Toperation* , matriz global recuperada
     */
    Toperation* mpiRecoverDistributedMatrixSendRecv(int root);
    /**
     * @brief Indica a que fila pertenece la matriz
     * 
     * @return int 
     */
    int getRowColor();
    /**
     * @brief Indica a que columna pertenece la matriz
     * 
     * @return int 
     */
    int getColumnColor();
    
};
#endif