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

void MpiMatrix::mpiSumma(int rowsA,int columnsAorRowsB,int columnsB,double* Ablock,double* Bblock,double* Cblock,int procGridrows,int procGridColumns)
{
    int i,indexFirstRow,indexFirstColumn;
    MPI_Group groupInitial, groupRow, groupColumn;
    MPI_Comm commRow, commCol;
    double *matrixLocalC= MatrixUtilities::matrixMemoryAllocation(blockNSize,blockNSize);
    double *bufferA;
    double *bufferB;
    int indexRowCnt = 0;
    int indexColCnt = 0;
    int localRowCnt = 0;
    int localColCnt = 0;
    int pb=blockSize;
    int m=rowsA;
    int n=columnsAorRowsB;
    int k= columnsB;
    int whoseTurnRow;
    int whoseTurnCol;
    int sizeStripA = pb * m/procGridrows;
    int sizeStripB = pb * n/procGridColumns;

    MPI_Comm_group(MPI_COMM_WORLD, &groupInitial);
    indexFirstRow = cpuRank % procGridrows;
    indexFirstColumn = (cpuRank - indexFirstRow)/procGridrows;//Seguro que es GridRows?
    // cout<< "Soy la cpu: "<<cpuRank<<" mi indexFirstRow es: "<< indexFirstRow << " y mi indexFirstColumn es: "<<indexFirstColumn<<endl;

    int cpuStrideGridColumn=cpuRank%procGridrows;
    int colGroupIndex[procGridColumns];
    int rowGroupIndex[procGridrows];
    for(i = 0; i< procGridrows; i++)
    {
        rowGroupIndex[i] = indexFirstColumn * procGridColumns + i;
        // printf("Soy el cpu %d y mi rowGroupIndex[%d] es: %d\n",cpuRank,i, rowGroupIndex[i]);
    }
    for(i = 0; i < procGridColumns; i++)
    {
        colGroupIndex[i] =i *procGridrows +indexFirstRow ;
        // printf("Soy el cpu %d y mi colGroupIndex[%d] es: %d\n",cpuRank,i, colGroupIndex[i]);
    }
    
    if(MPI_Group_incl(groupInitial, procGridColumns, rowGroupIndex, &groupRow) || MPI_Group_incl(groupInitial, procGridrows, colGroupIndex, &groupColumn))
    {
        cout<<"ERROR"<<endl;
    }
    if(MPI_Comm_create(MPI_COMM_WORLD, groupRow, &commRow) || MPI_Comm_create(MPI_COMM_WORLD, groupColumn, &commCol))
    {
        cout<<"ERROR"<<endl;
    }
    //Tantas iteraciones como bloques haya
    cout<<"El numero de iteraciones es: "<<columnsB/blockSize<<" las columnas de b son: "<< columnsB<<"tamaño del bloque: "<<blockSize<<endl;
    for(i=0;i<procGridrows*procGridColumns;i++)
    {

    }
    // for(i=0;i<columnsB/blockSize;++i)
    // {
    //     int panelRowCnt = 0;
    //     int panelColCnt = 0;

    //     int auxRowPb = pb;
    //     int auxColPb = pb;

    //     int nBlocksRow = pb / (k / procGridColumns);
    //     if(pb % (k / procGridColumns) > 0)
    //         ++nBlocksRow;

    //     int nBlocksCol = pb / (k / procGridrows);
    //     if(pb % (k / procGridrows) > 0)
    //         ++nBlocksCol;

    //     whoseTurnRow = (int) indexRowCnt / (k / procGridColumns);
    //     whoseTurnCol = (int) indexColCnt / (k / procGridrows);

        
    //     bufferA = (double *) malloc(sizeStripA * sizeof(double));
    //     bufferB = (double *) malloc(sizeStripB * sizeof(double));

    //     /* Rows */

    //     while(auxRowPb > 0)
    //     {
    //         int c;
    //         int lengthBand = std::min(k / procGridColumns - localRowCnt, auxRowPb);
    //         double *localBufferA = (double *) malloc(lengthBand * (m / procGridrows) * sizeof(double));

    //         whoseTurnRow = (int) indexRowCnt / (k / procGridColumns);

    //         if(indexFirstColumn == whoseTurnRow)
    //         {
    //             /* Fill local buffer */
    //             int l;

    //             /* Copy A's coefficients to buffer */
    //             for(l = 0; l < lengthBand * (m / procGridrows); ++l)
    //             {
    //                 localBufferA[l] = Ablock[localRowCnt * (m / procGridrows) + l]; 
    //             }

    //             if(MPI_Bcast(localBufferA, lengthBand * (m/ procGridrows), MPI_DOUBLE, whoseTurnRow, commRow))
    //             {
    //                 fprintf(stderr, "[Rank %d, i = %d] Error!", cpuRank, i);
    //                 MPI_Finalize();
    //             }

    //         }
    //         else
    //         {
    //             if(MPI_Bcast(localBufferA, lengthBand * (m/ procGridrows), MPI_DOUBLE, whoseTurnRow, commRow))
    //             {
    //                 fprintf(stderr, "[Rank %d, i = %d] Error!", cpuRank, i);
    //                 MPI_Finalize();
    //             }

    //         }


    //         /* Fill up bufferA */
    //         for(c = 0; c < lengthBand * (m / procGridrows); ++c)
    //         {
    //             bufferA[panelRowCnt * (m / procGridrows) + c] = localBufferA[c];

    //         }

    //         indexRowCnt += lengthBand;
    //         auxRowPb -= lengthBand;

    //         localRowCnt += lengthBand; 
    //         if(localRowCnt % (k / procGridColumns) == 0)
    //             localRowCnt = 0;

    //         panelRowCnt += lengthBand;
    //         if(panelRowCnt % pb == 0)
    //             panelRowCnt = 0;


    //         free(localBufferA);

    //     } /* End for(b) */


    //     /* Columns */

    //     while(auxColPb > 0)
    //     {
    //         int c;
    //         int r;
    //         int cnt = 0;
    //         int lengthBand = std::min(k / procGridrows - localColCnt, auxColPb);
    //         double *localBufferB = (double *) malloc(lengthBand * (n / procGridColumns) * sizeof(double));

    //         whoseTurnCol = (int) indexColCnt / (k / procGridrows);

    //         if(indexFirstRow == whoseTurnCol)
    //         {
    //             /* Fill local buffer */

    //             /* Copy B's coefficients to buffer */
    //             for(c = 0; c < n / procGridColumns; ++c)
    //             {
    //                 for(r = 0; r < lengthBand; ++r)
    //                 {
    //                     localBufferB[cnt] = Bblock[c * k / procGridrows  + localColCnt + r];
    //                     ++cnt;
    //                 }
    //             }

    //             if(MPI_Bcast(localBufferB, lengthBand * (n / procGridColumns), MPI_DOUBLE, whoseTurnCol, commCol))
    //             {
    //                 fprintf(stderr, "[Rank %d, i = %d] Error!", cpuRank, i);
    //                 MPI_Finalize();
    //             }

    //         }
    //         else
    //         {
    //             if(MPI_Bcast(localBufferB, lengthBand * (n / procGridColumns), MPI_DOUBLE, whoseTurnCol, commCol))
    //             {
    //                 fprintf(stderr, "[Rank %d, i = %d] Error!", cpuRank, i);
    //                 MPI_Finalize();
    //             }

    //         }


    //         /* Fill up bufferB */
    //         cnt = 0;
    //         for(c = 0; c < n / procGridColumns; ++c)
    //         {
    //             for(r = 0; r < lengthBand; ++r)
    //             {
    //                 bufferB[c * pb  + panelColCnt + r] = localBufferB[cnt];
    //                 ++cnt;
    //             }
    //         }

    //         indexColCnt += lengthBand;
    //         auxColPb -= lengthBand;

    //         localColCnt += lengthBand;
    //         if(localColCnt % (k / procGridrows) == 0)
    //             localColCnt = 0;

    //         panelColCnt += lengthBand;
    //         if(panelColCnt % pb == 0)
    //             panelColCnt = 0;

    //         free(localBufferB);

    //     } /* End for(b) */



    //     /* Multiply */

    //     // local_mm(blockNSize, blockNSize, pb, 1.0, bufferA,blockNSize, bufferB, pb, 1.0, Cblock, blockNSize);
    //     // Cblock=MatrixUtilities::matrixBlasMultiplication(blockNSize,blockNSize,pb,bufferA,bufferB,Cblock);
    //     cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,blockNSize,blockNSize,pb,1.0,bufferA,blockNSize,bufferB,pb,1.0,Cblock,blockNSize);
    //     printf("[Rank %d, i = %d] Local A: %f, local B: %f (Bblock[0]: %f). Result: %f\n", cpuRank, i, bufferA[0], bufferB[0], Bblock[0], Cblock[0]);


    //     free(bufferA);
    //     free(bufferB);
    // }
    
}


