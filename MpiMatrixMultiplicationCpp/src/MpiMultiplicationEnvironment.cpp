#include "MpiMultiplicationEnvironment.h"

template <class Toperation>
MpiMultiplicationEnvironment<Toperation>::MpiMultiplicationEnvironment(int cpuRank, int cpuSize,MPI_Comm commOperation)
{
    this->cpuRank = cpuRank;
    this->cpuSize = cpuSize;
    this->commOperation=commOperation;
    int i;
    for(i=0;i<cpuSize;i++)
    {

    }
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
    ////////////blockRowSize momentaneo, seguramente lo tenga que campiar
    int blockRowSizeA = matrixLocalA.getBlockRowSize();
    int blockColumnsSizeA = matrixLocalA.getBlockColumnSize();
    int blockColumnsSizeB = matrixLocalB.getBlockColumnSize();
    int blockRowSizeB = matrixLocalB.getBlockRowSize();
    int blockRowSize = matrixLocalA.getBlockRowSize();
    //////////////
    Toperation *matrixLocalC = MatrixUtilities<Toperation>::matrixMemoryAllocation(blockRowSizeA, blockColumnsSizeB);
    Toperation *matrixAuxiliarA = MatrixUtilities<Toperation>::matrixMemoryAllocation(blockRowSizeA, blockColumnsSizeA);
    Toperation *matrixAuxiliarB = MatrixUtilities<Toperation>::matrixMemoryAllocation(blockRowSizeB, blockColumnsSizeB);
    MPI_Comm_group(commOperation, &groupInitial);
    //Conseguir a que columna y fila pertenezco
    int rowColor=cpuRank/meshColumnsSize;
    int columnColor=cpuRank%meshColumnsSize;
    //Creacion de los nuevos grupos comunicadores para hacer Broadcast de filas o columnas a los pertenecientes a la malla de misma fila o columna
    usleep(2000);
    MPI_Barrier(commOperation);
    int colGroupIndex[meshColumnsSize];
    int rowGroupIndex[meshRowsSize];
    for (i = 0; i < meshColumnsSize; i++)
    {
        rowGroupIndex[i] = rowColor*meshColumnsSize +i;
        // printf("Soy el cpu %d y mi rowGroupIndex[%d] es: %d\n",cpuRank,i, rowGroupIndex[i]);
    }
    for (i = 0; i < meshRowsSize; i++)
    {
        colGroupIndex[i] = columnColor +i*meshColumnsSize;
        // printf("Soy el cpu %d y mi colGroupIndex[%d] es: %d\n",cpuRank,i, colGroupIndex[i]);
    }

    MPI_Group_incl(groupInitial, meshColumnsSize, rowGroupIndex, &groupRow);
    MPI_Group_incl(groupInitial, meshRowsSize, colGroupIndex, &groupColumn);
    MPI_Comm_create(commOperation, groupRow, &commRow);
    MPI_Comm_create(commOperation, groupColumn, &commCol);
    // std::cout<<"blockSizeA: "<<blockSizeA<<", blockSizeB: "<<blockSizeB<<std::endl;
    //Tantas iteraciones como bloques haya
    for (i = 0; i < meshRowsSize; i++)
    {
        MPI_Barrier(commOperation);
        // MatrixUtilities::debugMatrixDifferentCpus(cpuRank,blockRowSize,blockRowSize,matrixLocalC,".Inicio Iteracion: "+to_string(i));
        if (columnColor == i)
        {
            // cout<<"Soy la cpu: "<< cpuRank<<", y hago copia de mi A. Iteracion: "<<i<<endl;
            memcpy(matrixAuxiliarA, matrixLocalA.getMatrixLocal(), blockSizeA * sizeof(Toperation));
        }
        if (rowColor == i)
        {
            // cout<<"Soy la cpu: "<< cpuRank<<", y hago copia de mi B. Iteracion: "<<i<<endl;
            memcpy(matrixAuxiliarB, matrixLocalB.getMatrixLocal(), blockSizeB * sizeof(Toperation));
        }
        MPI_Bcast(matrixAuxiliarA, blockSizeA, MPI_DOUBLE, i, commRow);
        MPI_Bcast(matrixAuxiliarB, blockSizeB, MPI_DOUBLE, i, commCol);
        //Habra que cambiar los blockRowSize para cuando se hagan mas procesos
        MatrixUtilities<Toperation>::Multiplicacion(blockRowSizeA,blockRowSizeB,blockColumnsSizeB,matrixAuxiliarA,matrixAuxiliarB,matrixLocalC);
        // MatrixUtilities::matrixBlasMultiplication(blockRowSizeA, blockRowSizeB, blockColumnsSizeB, matrixAuxiliarA, matrixAuxiliarB, matrixLocalC);
        // if (cpuRank == 0)
        // {
        //     std::cout<<"blockRowSizeA: "<<blockRowSizeA<<", blockColumnsSizeA: "<<blockColumnsSizeA<<", blockRowSizeB: "<<blockRowSizeB<<", blockColumnsSizeB: "<<blockColumnsSizeB<<std::endl;
        // }
        // MatrixUtilities::debugMatrixDifferentCpus(cpuRank, blockRowSize, blockRowSize, matrixLocalC, ".Final Iteracion: " + std::to_string(i));
    }
    MpiMatrix<Toperation> res = MpiMatrix<Toperation>(cpuSize, cpuRank, meshRowsSize, meshColumnsSize, rowsA, columnsB,commOperation);
    res.setMatrixLocal(matrixLocalC);
    return res;
}



template class MpiMultiplicationEnvironment<double>;
template class MpiMultiplicationEnvironment<float>;
template class MpiMultiplicationEnvironment<int>;


