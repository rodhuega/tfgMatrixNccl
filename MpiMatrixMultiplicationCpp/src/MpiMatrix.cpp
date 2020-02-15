#include "MpiMatrix.h"


MpiMatrix::MpiMatrix(int cpuSize, int cpuRank, int meshRowSize, int meshColumnSize, int rowSize, int columnSize)
{
    this->cpuRank = cpuRank;
    this->cpuSize = cpuSize;
    this->meshRowSize = meshRowSize;
    this->meshColumnSize = meshColumnSize;
    this->rowSize = rowSize;
    this->columnSize = columnSize;
    blockRowSize = rowSize / meshRowSize;
    blockColumnSize = columnSize / meshColumnSize;
    blockSize = blockRowSize * blockColumnSize;
    sendCounts.reserve(cpuSize);
    std::fill_n(sendCounts.begin(), cpuSize, 1);
    int i, posColumnBelong, posRowBelong;
    //Asignamos en que posicion empieza cada bloque
    for (i = 0; i < cpuSize; i++)
    {
        posRowBelong = (i / meshColumnSize) * columnSize * blockRowSize;
        posColumnBelong = (i % meshColumnSize) * blockColumnSize;
        blocks.push_back(posColumnBelong + posRowBelong);
    }
    if (cpuRank == 0)
    {
        int sizes[2] = {rowSize, columnSize};
        int subsizes[2] = {blockRowSize, blockColumnSize};
        int starts[2] = {0, 0};
        MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &matrixLocalType);
        int doubleSize;
        MPI_Type_size(MPI_DOUBLE, &doubleSize);
        MPI_Type_create_resized(matrixLocalType, 0, 1 * doubleSize, &matrixLocalType);
        MPI_Type_commit(&matrixLocalType);
    }
}

int MpiMatrix::getBlockRowSize()
{
    return blockRowSize;
}

int MpiMatrix::getBlockColumnSize()
{
    return blockColumnSize;
}

double *MpiMatrix::mpiDistributeMatrix(double *matrixGlobal, int root)
{
    double *globalptr = NULL;
    if (cpuRank == root)
    {
        globalptr = matrixGlobal;
    }
    double *matrixLocal = MatrixUtilities::matrixMemoryAllocation(blockRowSize, blockColumnSize);
    MPI_Scatterv(globalptr, &sendCounts[0], &blocks[0], matrixLocalType, matrixLocal, blockSize, MPI_DOUBLE, root, MPI_COMM_WORLD);
    return matrixLocal;
}

double *MpiMatrix::mpiRecoverDistributedMatrixGatherV(double *matrixLocal, int root)
{
    double *matrix = NULL;
    if (cpuRank == root)
    {
        matrix = MatrixUtilities::matrixMemoryAllocation(rowSize, columnSize);
    }
    MPI_Gatherv(matrixLocal, blockSize, MPI_DOUBLE, matrix, &sendCounts[0], &blocks[0], matrixLocalType, root, MPI_COMM_WORLD);
    // if(cpuRank==0) WIP: DESTRUCTOR
    // {
    //     MPI_Type_free(&matrixLocalType);
    // }
    return matrix;
}

//CREO QUE FALLA CON MATRICES NO CUADRADAS
double *MpiMatrix::mpiRecoverDistributedMatrixReduce(double *matrixLocal, int root)
{
    double *matrix = NULL;
    int i;
    double *matrixLocalTotalNSize = MatrixUtilities::matrixMemoryAllocation(rowSize, columnSize);
    int initialBlockPosition = blocks[cpuRank];

    if (cpuRank == root)
    {
        matrix = MatrixUtilities::matrixMemoryAllocation(rowSize, columnSize);
    }

    for (i = 0; i < blockRowSize; i++)
    {
        memcpy(&matrixLocalTotalNSize[initialBlockPosition + i * columnSize], &matrixLocal[i * blockColumnSize], blockColumnSize * sizeof(double));
    }
    MPI_Reduce(matrixLocalTotalNSize, matrix, rowSize * columnSize, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
    return matrix;
}

double *MpiMatrix::mpiSumma(int rowsA, int columnsAorRowsB, int columnsB, double *matrixLocalA, double *matrixLocalB, int meshRowsSize, int meshColumnsSize)
{
    int i;
    MPI_Group groupInitial, groupRow, groupColumn;
    MPI_Comm commRow, commCol;
    double *matrixLocalC = MatrixUtilities::matrixMemoryAllocation(blockRowSize, blockRowSize);
    double *matrixAuxiliarA = MatrixUtilities::matrixMemoryAllocation(blockRowSize, blockRowSize);
    double *matrixAuxiliarB = MatrixUtilities::matrixMemoryAllocation(blockRowSize, blockRowSize);
    int sizeStripA = blockSize * rowsA / meshRowsSize;
    int sizeStripB = blockSize * columnsAorRowsB / meshColumnsSize;

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
            memcpy(matrixAuxiliarA, matrixLocalA, blockSize * sizeof(double));
        }
        if (cpuRank / meshRowsSize == i)
        {
            memcpy(matrixAuxiliarB, matrixLocalB, blockSize * sizeof(double));
        }
        MPI_Bcast(matrixAuxiliarA, blockSize, MPI_DOUBLE, i, commRow);
        MPI_Bcast(matrixAuxiliarB, blockSize, MPI_DOUBLE, i, commCol);
        MatrixUtilities::matrixBlasMultiplication(blockRowSize, blockRowSize, blockRowSize, matrixAuxiliarA, matrixAuxiliarB, matrixLocalC);
        // MatrixUtilities::debugMatrixDifferentCpus(cpuRank,blockRowSize,blockRowSize,matrixLocalC,"Final Iteracion: "+to_string(i));
    }
    return matrixLocalC;
}