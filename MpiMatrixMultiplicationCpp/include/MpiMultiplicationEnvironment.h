#ifndef MpiMultiplicationEnvironment_H
#define MpiMultiplicationEnvironment_H

#include "MatrixUtilities.h"
#include "MpiMatrix.h"

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
    int cpuRank,cpuSize;
    

public:
    /**
     * @brief Se construye el objeto que realizara la multiplicacion distribuida mediante el algoritmo suma
     * 
     * @param cpuRank , Id de la cpu
     * @param cpuSize , Numero de procesadores que realizaran el computo
     * @param commOperation , Comunicador de los procesadores que realizaran el computo
     * @param basicOperationType Tipo de numero con el que se realizara la multiplicacion(Double, Int..)
     */
    MpiMultiplicationEnvironment(int cpuRank,int cpuSize,MPI_Comm commOperation,MPI_Datatype basicOperationType);
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