#include "MpiMatrix.h"
#include "MatrixUtilities.h"
#include <unistd.h>
#include <cblas.h>

MpiMatrix::MpiMatrix(int cpuSize,int cpuRank, int NSize)
{
    this->cpuRank = cpuRank;
    this->cpuSize=cpuSize;
    N = NSize;
    blockNSize = N / 2;
    blockSize = blockNSize * blockNSize;
    sendCounts.reserve(cpuSize);
    std::fill_n(sendCounts.begin(),cpuSize,1);
    //WIP: MAS PROCESOS
    blocks.push_back(0);
    blocks.push_back(blockNSize);
    blocks.push_back(N * blockNSize);
    blocks.push_back( N * blockNSize + blockNSize);
    if (cpuRank == 0)
    {
        int sizes[2] = {N, N};
        int subsizes[2] = {blockNSize, blockNSize};
        int starts[2] = {0, 0};
        MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &matrixLocalType);
        int doubleSize;
        MPI_Type_size(MPI_DOUBLE, &doubleSize);
        MPI_Type_create_resized(matrixLocalType, 0, 1 * doubleSize, &matrixLocalType);
        MPI_Type_commit(&matrixLocalType);
    }
}

double *MpiMatrix::mpiDistributeMatrix(double *matrixGlobal, int root)
{
    double *globalptr = NULL;
    if (cpuRank == root)
    {
        globalptr = matrixGlobal;
    }
    //WIP: MAS PROCESOS
    int matrixLocalIndices[4] = {blocks[0], blocks[1], blocks[2], blocks[3]};
    double *matrixLocal = MatrixUtilities::matrixMemoryAllocation(blockNSize,blockNSize);
    MPI_Scatterv(globalptr, &sendCounts[0], matrixLocalIndices, matrixLocalType, matrixLocal, blockSize, MPI_DOUBLE, root, MPI_COMM_WORLD);
    return matrixLocal;
}

double *MpiMatrix::mpiRecoverDistributedMatrixGatherV(double *matrixLocal, int root)
{
    double *matrix = NULL;
    if (cpuRank == root)
    {
        matrix = MatrixUtilities::matrixMemoryAllocation(N,N);
    }
    MPI_Gatherv(matrixLocal, blockSize, MPI_DOUBLE, matrix, &sendCounts[0], &blocks[0], matrixLocalType, root, MPI_COMM_WORLD);
    // if(cpuRank==0) WIP: DESTRUCTOR
    // {
    //     MPI_Type_free(&matrixLocalType);
    // }
    return matrix;
}

double *MpiMatrix::mpiRecoverDistributedMatrixReduce(double *matrixLocal, int root)
{
    double *matrix = NULL;
    int i;
    double *matrixLocalTotalNSize = MatrixUtilities::matrixMemoryAllocation(N,N);
    int initialBlockPosition = blocks[cpuRank];

    if (cpuRank == root)
    {
        matrix = MatrixUtilities::matrixMemoryAllocation(N,N);
    }
    
    for (i = 0; i < blockNSize; i++)
    {
        memcpy(&matrixLocalTotalNSize[initialBlockPosition + i * N], &matrixLocal[i * blockNSize], blockNSize * sizeof(double));
    }
    MPI_Reduce(matrixLocalTotalNSize,matrix,N*N,MPI_DOUBLE,MPI_SUM,root,MPI_COMM_WORLD);
    return matrix;
}

double* MpiMatrix::mpiSumma(int rowsA,int columnsAorRowsB,int columnsB,double* matrixLocalA,double* matrixLocalB,int gridRows,int gridColumns)
{
    int i;
    MPI_Group groupInitial, groupRow, groupColumn;
    MPI_Comm commRow, commCol;
    double *matrixLocalC= MatrixUtilities::matrixMemoryAllocation(blockNSize,blockNSize);
    double *matrixAuxiliarA=MatrixUtilities::matrixMemoryAllocation(blockNSize,blockNSize);
    double *matrixAuxiliarB=MatrixUtilities::matrixMemoryAllocation(blockNSize,blockNSize);
    int sizeStripA = blockSize * rowsA/gridRows;
    int sizeStripB = blockSize * columnsAorRowsB/gridColumns;

    MPI_Comm_group(MPI_COMM_WORLD, &groupInitial);
    int indexFirstRow = cpuRank % gridRows;
    int indexFirstColumn = (cpuRank - indexFirstRow)/gridRows;//Seguro que es GridRows?
    // cout<< "Soy la cpu: "<<cpuRank<<" mi indexFirstRow es: "<< indexFirstRow << " y mi indexFirstColumn es: "<<indexFirstColumn<<endl;

    int cpuStrideGridColumn=cpuRank%gridRows;
    int colGroupIndex[gridColumns];
    int rowGroupIndex[gridRows];
    for(i = 0; i< gridRows; i++)
    {
        rowGroupIndex[i] = indexFirstColumn * gridColumns + i;
        // printf("Soy el cpu %d y mi rowGroupIndex[%d] es: %d\n",cpuRank,i, rowGroupIndex[i]);
    }
    for(i = 0; i < gridColumns; i++)
    {
        colGroupIndex[i] =i *gridRows +indexFirstRow ;
        // printf("Soy el cpu %d y mi colGroupIndex[%d] es: %d\n",cpuRank,i, colGroupIndex[i]);
    }
    
    if(MPI_Group_incl(groupInitial, gridColumns, rowGroupIndex, &groupRow) || MPI_Group_incl(groupInitial, gridRows, colGroupIndex, &groupColumn))
    {
        cout<<"ERROR"<<endl;
    }
    if(MPI_Comm_create(MPI_COMM_WORLD, groupRow, &commRow) || MPI_Comm_create(MPI_COMM_WORLD, groupColumn, &commCol))
    {
        cout<<"ERROR"<<endl;
    }
    //Tantas iteraciones como bloques haya
    for(i=0;i<gridRows;i++)
    {
        // MatrixUtilities::debugMatrixDifferentCpus(cpuRank,blockNSize,blockNSize,matrixLocalC,"Inicio Iteracion: "+to_string(i));
        if(cpuRank%gridRows==i)
        {
            memcpy(matrixAuxiliarA,matrixLocalA,blockSize*sizeof(double));
        }
        if(cpuRank/gridRows==i)
        {
            memcpy(matrixAuxiliarB,matrixLocalB,blockSize*sizeof(double));
        }
        MPI_Bcast(matrixAuxiliarA,blockSize,MPI_DOUBLE,i,commRow);
        MPI_Bcast(matrixAuxiliarB,blockSize,MPI_DOUBLE,i,commCol);
        MatrixUtilities::matrixBlasMultiplication(blockNSize,blockNSize,blockNSize,matrixAuxiliarA,matrixAuxiliarB,matrixLocalC);
        // MatrixUtilities::debugMatrixDifferentCpus(cpuRank,blockNSize,blockNSize,matrixLocalC,"Final Iteracion: "+to_string(i));
    }
    return matrixLocalC;
}