#include "MpiMultiplicationEnvironment.h"

MpiMultiplicationEnvironment::MpiMultiplicationEnvironment(int cpuRank, int cpuSize)
{
    this->cpuRank=cpuRank;
    this->cpuSize=cpuSize;
}

MpiMatrix MpiMultiplicationEnvironment::mpiSumma(MpiMatrix matrixLocalA, MpiMatrix matrixLocalB, int meshRowsSize, int meshColumnsSize)
{
    int i;
    MPI_Group groupInitial, groupRow, groupColumn;
    MPI_Comm commRow, commCol;
    int rowsA=matrixLocalA.getRowSize();
    int columnsAorRowsB=matrixLocalA.getColumnSize();
    int columnsB=matrixLocalB.getColumnSize();
    int blockSizeA=matrixLocalA.getBlockSize();
    int blockSizeB=matrixLocalB.getBlockSize();
    ////////////blockRowSize momentaneo, seguramente lo tenga que campiar
    int blockRowSize=matrixLocalA.getBlockRowSize();
    //////////////
    double *matrixLocalC = MatrixUtilities::matrixMemoryAllocation(blockRowSize, blockRowSize);
    double *matrixAuxiliarA = MatrixUtilities::matrixMemoryAllocation(blockRowSize, blockRowSize);
    double *matrixAuxiliarB = MatrixUtilities::matrixMemoryAllocation(blockRowSize, blockRowSize);
    int sizeStripA = blockSizeA * rowsA / meshRowsSize;
    int sizeStripB = blockSizeB * columnsAorRowsB / meshColumnsSize;

    MPI_Comm_group(MPI_COMM_WORLD, &groupInitial);
    int indexFirstRow = cpuRank % meshRowsSize;
    int indexFirstColumn = (cpuRank - indexFirstRow) / meshRowsSize; //Seguro que es GridRows?
    // cout<< "Soy la cpu: "<<cpuRank<<" mi indexFirstRow es: "<< indexFirstRow << " y mi indexFirstColumn es: "<<indexFirstColumn<<endl;

    int cpuStrideGridColumn = cpuRank % meshRowsSize;
    int colGroupIndex[meshColumnsSize];
    int rowGroupIndex[meshRowsSize];
    for (i = 0; i < meshRowsSize; i++)
    {
        rowGroupIndex[i] = indexFirstColumn * meshColumnsSize + i;
        // printf("Soy el cpu %d y mi rowGroupIndex[%d] es: %d\n",cpuRank,i, rowGroupIndex[i]);
    }
    for (i = 0; i < meshColumnsSize; i++)
    {
        colGroupIndex[i] = i * meshRowsSize + indexFirstRow;
        // printf("Soy el cpu %d y mi colGroupIndex[%d] es: %d\n",cpuRank,i, colGroupIndex[i]);
    }

    if (MPI_Group_incl(groupInitial, meshColumnsSize, rowGroupIndex, &groupRow) || MPI_Group_incl(groupInitial, meshRowsSize, colGroupIndex, &groupColumn))
    {
        cout << "ERROR" << endl;
    }
    if (MPI_Comm_create(MPI_COMM_WORLD, groupRow, &commRow) || MPI_Comm_create(MPI_COMM_WORLD, groupColumn, &commCol))
    {
        cout << "ERROR" << endl;
    }
    //Tantas iteraciones como bloques haya
    for (i = 0; i < meshRowsSize; i++)
    {
        // MatrixUtilities::debugMatrixDifferentCpus(cpuRank,blockRowSize,blockRowSize,matrixLocalC,"Inicio Iteracion: "+to_string(i));
        if (cpuRank % meshRowsSize == i)
        {
            memcpy(matrixAuxiliarA, matrixLocalA.getMatrixLocal(), blockSizeA * sizeof(double));
        }
        if (cpuRank / meshRowsSize == i)
        {
            memcpy(matrixAuxiliarB, matrixLocalB.getMatrixLocal(), blockSizeB * sizeof(double));
        }
        MPI_Bcast(matrixAuxiliarA, blockSizeA, MPI_DOUBLE, i, commRow);
        MPI_Bcast(matrixAuxiliarB, blockSizeB, MPI_DOUBLE, i, commCol);
        MatrixUtilities::matrixBlasMultiplication(blockRowSize, blockRowSize, blockRowSize, matrixAuxiliarA, matrixAuxiliarB, matrixLocalC);
        // MatrixUtilities::debugMatrixDifferentCpus(cpuRank,blockRowSize,blockRowSize,matrixLocalC,"Final Iteracion: "+to_string(i));
    }
    MpiMatrix res = MpiMatrix(cpuSize,cpuRank,meshRowsSize,meshColumnsSize,rowsA,columnsB);
    res.setMatrixLocal(matrixLocalC);
    return res;
}