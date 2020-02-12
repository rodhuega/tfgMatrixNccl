#include <string>
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include "MatrixMain.h"
#include "MatrixBlock.h"
#include "MpiMatrix.h"

using namespace std;

void PrintDoublePointerMatrix(int rows, int columns, double **M)
{
    int i, j;
    for (i = 0; i < rows; ++i)
    {
        for (j = 0; j < columns; ++j)
        {
            cout << M[i][j] << "\t";
        }
        cout << endl;
    }
}

void PrintOnePointerMatrix(int rows, int columns, double *M)
{
    int i, j;
    for (i = 0; i < rows; ++i)
    {
        for (j = 0; j < columns; ++j)
        {
            cout << M[i * rows + j] << "\t";
        }
        cout << endl;
    }
}

void matrixMultiplication(int N, double *A, double *B, double *C)
{
    int i, j, k, sum;
    for (i = 0; i < N; i++)
    {
        sum = 0;
        for (j = 0; j < N; j++)
        {
            sum = 0;
            for (k = 0; k < N; k++)
            {
                sum += A[i * N + j] * B[i + N * j];
            }
            C[i*N+j]= sum;
        }
    }
}

// void copiarMiniMatriz(int primeraFila,int primeraColumna,double **A,double* a_local)
// {
//     int i;
//     for(i=0;i<blockNSize;i++)
//     {
//         memcpy(a_local+i*blockNSize, &A[primeraFila][primeraColumna], sizeof(double)*blockNSize);
//         primeraFila+=1;
//     }
// }

// void RealizarMultiplicacionBloque(int cpuRank, int blockNumberOfElements, double **A, double **B, double *c_local)
// {
//     int i, j;
//     //Realizamos las dos multiplicaciones necesarias
//     double a1_local[blockNSize][blockNSize];
//     double a2_local[blockNSize][blockNSize];
//     double b1_local[blockNSize][blockNSize];
//     double b2_local[blockNSize][blockNSize];
//     double c1_local[blockNSize][blockNSize];
//     double c2_local[blockNSize][blockNSize];
//     if (cpuRank == 0)
//     {
//         copiarMiniMatriz(0,0,A,a1_local);
//         copiarMiniMatriz(0,blockNSize,A,a2_local);
//         copiarMiniMatriz(0,0,B,b1_local);
//         copiarMiniMatriz(blockNSize,0,B,b2_local);
//     }
//     if (cpuRank == 1)
//     {
//         copiarMiniMatriz(0,0,A,a1_local);
//         copiarMiniMatriz(0,blockNSize,A,a2_local);
//         copiarMiniMatriz(0,blockNSize,B,b1_local);
//         copiarMiniMatriz(blockNSize,blockNSize,B,b2_local);
//     }
//     if (cpuRank == 2)
//     {
//         copiarMiniMatriz(blockNSize,0,A,a1_local);
//         copiarMiniMatriz(blockNSize,blockNSize,A,a2_local);
//         copiarMiniMatriz(0,0,B,b1_local);
//         copiarMiniMatriz(blockNSize,0,B,b2_local);
//     }
//     if (cpuRank == 3)
//     {
//         copiarMiniMatriz(blockNSize,0,A,a1_local);
//         copiarMiniMatriz(blockNSize,blockNSize,A,a2_local);
//         copiarMiniMatriz(0,blockNSize,B,b1_local);
//         copiarMiniMatriz(blockNSize,blockNSize,B,b2_local);
//     }
//     RealizarMultiplicacion(cpuRank, a1_local, b1_local, c1_local);
//     RealizarMultiplicacion(cpuRank, a2_local, b2_local, c2_local);
//     //Sumamos los productos de esas multiplicaciones
//     for (i = 0; i < blockNSize; i++)
//     {
//         for (j = 0; j < blockNSize; j++)
//         {
//             c_local[blockNSize * i + j] = c1_local[i][j] + c2_local[i][j];
//         }
//     }

// }
// double *mpiDistributeMatrix(int cpuRank, int N, double *matrixGlobal)
// {
//     double *globalptr = NULL;
    
//     if (cpuRank == 0)
//     {
//         globalptr = matrixGlobal;
//     }
//     int blockNSize = N / 2;
//     int blockSize = blockNSize * blockNSize;
//     MPI_Datatype matrixLocalType;
//     if (cpuRank == 0)
//     {
//         int sizes[2] = {N, N};
//         int subsizes[2] = {blockNSize, blockNSize};
//         int starts[2] = {0, 0};
//         MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &matrixLocalType);
//         int doubleSize;
//         MPI_Type_size(MPI_DOUBLE, &doubleSize);
//         MPI_Type_create_resized(matrixLocalType, 0, 1 * doubleSize, &matrixLocalType);
//         MPI_Type_commit(&matrixLocalType);
//     }

//     const int blocks[4] = {0, blockNSize, N * blockNSize, N * blockNSize + blockNSize};

//     int sendCounts[4] = {1, 1, 1, 1};
//     int matrixLocalIndices[4] = {blocks[0], blocks[1], blocks[2], blocks[3]};
//     double *matrixLocal = (double *)calloc(blockNSize * blockNSize, sizeof(double));
//     MPI_Scatterv(globalptr, sendCounts, matrixLocalIndices, matrixLocalType, matrixLocal, blockSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
// }

// double* mpiRecoverDistributedMatrixGatherV(int N,int cpuRank,double* localMatrix)
// {
//     int blockNSize = N / 2;
//     int blockSize = blockNSize * blockNSize;
//     double *matrix=(double*)calloc(N*N,sizeof(double));
//     MPI_Datatype matrixLocalType;
//     if (cpuRank == 0)
//     {
//         int sizes[2] = {N, N};
//         int subsizes[2] = {blockNSize, blockNSize};
//         int starts[2] = {0, 0};
//         MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &matrixLocalType);
//         int doubleSize;
//         MPI_Type_size(MPI_DOUBLE, &doubleSize);
//         MPI_Type_create_resized(matrixLocalType, 0, 1 * doubleSize, &matrixLocalType);
//         MPI_Type_commit(&matrixLocalType);
//     }

//     const int blocks[4] = {0, blockNSize, N * blockNSize, N * blockNSize + blockNSize};

//     int sendCounts[4] = {1, 1, 1, 1};
//     MPI_Gatherv(localMatrix, blockSize, MPI_DOUBLE, matrix, sendCounts, blocks, matrixLocalType, 0, MPI_COMM_WORLD);
//     if(cpuRank==0)
//     {
//         MPI_Type_free(&matrixLocalType);
//         PrintOnePointerMatrix(N,N,matrix);
//     }
// }

int main(int argc, char *argv[])
{
    int cpuRank, cpuSize;
    int rowsA, columnsA, rowsB, columnsB;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &cpuSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &cpuRank);
    double *a = NULL;
    if (cpuRank == 0)
    {

        MatrixMain ma = MatrixMain(argv[1]);
        a = ma.getMatrix();
        rowsA = ma.getRowsUsed();
        columnsA = ma.getColumnsUsed();
        cout << "La matriz A:" << endl;
        PrintOnePointerMatrix(rowsA, columnsA, a);
        cout << "Procedemos a distribuir A:" << endl;
    }
    MPI_Bcast(&rowsA, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MpiMatrix mMpiLocal= MpiMatrix(cpuRank,rowsA);
    double* localMatrix=mMpiLocal.mpiDistributeMatrix(a);
    usleep(cpuRank*1000);
    cout<<"Parte del proceso: "<<cpuRank<<endl;
    PrintOnePointerMatrix(rowsA/2,rowsA/2,localMatrix);
    cout<<endl;
    // mpiRecoverDistributedMatrixGatherV(rowsA,cpuRank,localMatrix);
    MPI_Finalize();
}
