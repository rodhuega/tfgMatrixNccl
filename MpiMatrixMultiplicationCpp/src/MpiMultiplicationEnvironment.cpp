#include "MpiMultiplicationEnvironment.h"

template <class Toperation>
MpiMultiplicationEnvironment<Toperation>::MpiMultiplicationEnvironment(int cpuRank, int cpuSize,MPI_Comm commOperation,MPI_Datatype basicOperationType)
{
    this->cpuRank = cpuRank;
    this->cpuSize = cpuSize;
    this->commOperation=commOperation;
    this->basicOperationType=basicOperationType;
}


template<class Toperation>
void MpiMultiplicationEnvironment<Toperation>::setNewMatrixGlobal(std::string id,MatrixMain<Toperation>* matrixMainGlobal)
{
    matricesGlobal[id]=matrixMainGlobal;
}

template<class Toperation>
MatrixMain<Toperation>* MpiMultiplicationEnvironment<Toperation>::getAMatrixGlobal(std::string id)
{
    auto it = matricesGlobal.find(id);
    if(it==matricesGlobal.end())
    {
        throw std::invalid_argument("La matriz no existe");
    }
    return it->second;
}

template <class Toperation>
MpiMatrix<Toperation> MpiMultiplicationEnvironment<Toperation>::mpiSumma(MpiMatrix<Toperation> matrixLocalA, MpiMatrix<Toperation> matrixLocalB, int meshRowsSize, int meshColumnsSize)
{
    int i;
    MPI_Group groupInitial, groupRow, groupColumn;
    MPI_Comm commRow, commCol;
    int rowsA = matrixLocalA.getRowSize();
    int columnsAorRowsB = matrixLocalA.getColumnSize();
    int columnsB = matrixLocalB.getColumnSize();
    int blockSizeA = matrixLocalA.getBlockSize();
    int blockSizeB = matrixLocalB.getBlockSize();
    int blockRowSizeA = matrixLocalA.getBlockRowSize();
    int blockColumnsSizeA = matrixLocalA.getBlockColumnSize();
    int blockColumnsSizeB = matrixLocalB.getBlockColumnSize();
    int blockRowSizeB = matrixLocalB.getBlockRowSize();
    int blockRowSize = matrixLocalA.getBlockRowSize();
    Toperation *matrixLocalC = MatrixUtilities<Toperation>::matrixMemoryAllocation(blockRowSizeA, blockColumnsSizeB);
    Toperation *matrixAuxiliarA = MatrixUtilities<Toperation>::matrixMemoryAllocation(blockRowSizeA, blockColumnsSizeA);
    Toperation *matrixAuxiliarB = MatrixUtilities<Toperation>::matrixMemoryAllocation(blockRowSizeB, blockColumnsSizeB);
    MPI_Comm_group(commOperation, &groupInitial);
    //Conseguir a que columna y fila pertenezco
    int rowColor=cpuRank/meshColumnsSize;
    int columnColor=cpuRank%meshColumnsSize;
    //Creacion de los nuevos grupos comunicadores para hacer Broadcast de filas o columnas a los pertenecientes a la malla de misma fila o columna
    int colGroupIndex[meshColumnsSize];
    int rowGroupIndex[meshRowsSize];
    for (i = 0; i < meshColumnsSize; i++)
    {
        rowGroupIndex[i] = rowColor*meshColumnsSize +i;
    }
    for (i = 0; i < meshRowsSize; i++)
    {
        colGroupIndex[i] = columnColor +i*meshColumnsSize;
    }
    MPI_Group_incl(groupInitial, meshColumnsSize, rowGroupIndex, &groupRow);
    MPI_Group_incl(groupInitial, meshRowsSize, colGroupIndex, &groupColumn);
    MPI_Comm_create(commOperation, groupRow, &commRow);
    MPI_Comm_create(commOperation, groupColumn, &commCol);
    //Realizacion de las operaciones matematicas
    for (i = 0; i < meshRowsSize; i++)
    {
        // MPI_Barrier(commOperation);
        // MatrixUtilities::debugMatrixDifferentCpus(cpuRank,blockRowSize,blockRowSize,matrixLocalC,".Inicio Iteracion: "+to_string(i));
        if (columnColor == i)
        {
            memcpy(matrixAuxiliarA, matrixLocalA.getMatrixLocal(), blockSizeA * sizeof(Toperation));
        }
        if (rowColor == i)
        {
            memcpy(matrixAuxiliarB, matrixLocalB.getMatrixLocal(), blockSizeB * sizeof(Toperation));
        }
        MPI_Bcast(matrixAuxiliarA, blockSizeA, basicOperationType, i, commRow);
        MPI_Bcast(matrixAuxiliarB, blockSizeB, basicOperationType, i, commCol);
        MatrixUtilities<Toperation>::Multiplicacion(blockRowSizeA,blockRowSizeB,blockColumnsSizeB,matrixAuxiliarA,matrixAuxiliarB,matrixLocalC);
        // MatrixUtilities::matrixBlasMultiplication(blockRowSizeA, blockRowSizeB, blockColumnsSizeB, matrixAuxiliarA, matrixAuxiliarB, matrixLocalC);
        // MatrixUtilities::debugMatrixDifferentCpus(cpuRank, blockRowSize, blockRowSize, matrixLocalC, ".Final Iteracion: " + std::to_string(i));
    }
    //Liberacion de las matrices auxiliares que realizaban computo
    MatrixUtilities<Toperation>::matrixFree(matrixAuxiliarA);
    MatrixUtilities<Toperation>::matrixFree(matrixAuxiliarB);
    //Creacion del objeto local que contiene el resultado local de la operacion y asignacion del resultado a este objeto
    MpiMatrix<Toperation> res = MpiMatrix<Toperation>(cpuSize, cpuRank, meshRowsSize, meshColumnsSize, rowsA, columnsB,commOperation,basicOperationType);
    res.setMatrixLocal(matrixLocalC);
    return res;
}



template class MpiMultiplicationEnvironment<double>;
template class MpiMultiplicationEnvironment<float>;


