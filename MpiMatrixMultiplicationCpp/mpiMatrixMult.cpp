#include <string>
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include "MatrixMain.h"
#include "MatrixBlock.h"
#include "MpiMatrix.h"
#include "MatrixUtilities.h"

using namespace std;



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
        // cout << "La matriz A:" << endl;
        // MatrixUtilities::PrintOnePointerMatrix(rowsA, columnsA, a);
        // cout << "Procedemos a distribuir A:" << endl;
    }
    MPI_Bcast(&rowsA, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MpiMatrix mMpiLocal= MpiMatrix(cpuSize,cpuRank,rowsA);
    double* localMatrix=mMpiLocal.mpiDistributeMatrix(a,0);
    usleep(cpuRank*1000);
    cout<<"Parte del proceso: "<<cpuRank<<endl;
    MatrixUtilities::PrintOnePointerMatrix(rowsA/2,rowsA/2,localMatrix);
    cout<<endl;
    double* matrixRecovered=mMpiLocal.mpiRecoverDistributedMatrixReduce(localMatrix,0);
    // double* matrixRecovered=mMpiLocal.mpiRecoverDistributedMatrixGatherV(localMatrix,0);
    if(cpuRank==0)
    {
        usleep(2000);
        cout<<"Matrix recuperada: "<<endl;
        MatrixUtilities::PrintOnePointerMatrix(rowsA,rowsA,matrixRecovered);
    }
    MPI_Finalize();
}
