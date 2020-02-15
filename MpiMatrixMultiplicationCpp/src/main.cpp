#include <string>
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <iomanip>
#include <math.h>
#include "MatrixMain.h"
#include "MatrixBlock.h"
#include "MpiMatrix.h"
#include "MatrixUtilities.h"

using namespace std;

int main(int argc, char *argv[])
{
    int cpuRank, cpuSize;
    int rowsA, columnsA, rowsB, columnsB;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &cpuSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &cpuRank);
    //Encontrar formula de calculo para estos parametros
    int meshRowColumnSize=sqrt(cpuSize);
    int meshRowSize=meshRowColumnSize;
    int meshColumnSize=meshRowColumnSize;
    //////////////
    cout<<fixed;
    cout<<setprecision(2);
    double *a = NULL;
    double *b = NULL;
    if (cpuRank == 0)
    {
        MatrixMain ma = MatrixMain(argv[1]);
        a = ma.getMatrix();
        rowsA = ma.getRowsUsed();
        columnsA = ma.getColumnsUsed();
        MatrixMain mb = MatrixMain(argv[2]);
        b = mb.getMatrix();
        rowsB = mb.getRowsUsed();
        columnsB = mb.getColumnsUsed();
        // cout << "La matriz A:" << endl;
        // MatrixUtilities::printMatrix(rowsA, columnsA, a);
        // cout << "Procedemos a distribuir A:" << endl;
        // cout << "La matriz B:" << endl;
        // MatrixUtilities::printMatrix(rowsB, columnsB, b);
        // cout << "Procedemos a distribuir B:" << endl;
    }
    MPI_Bcast(&rowsA, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&columnsA, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rowsB, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&columnsB, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MpiMatrix mMpiLocal= MpiMatrix(cpuSize,cpuRank,meshRowSize,meshColumnSize,rowsA,columnsA);
    double* aLocalMatrix=mMpiLocal.mpiDistributeMatrix(a,0);
    double* bLocalMatrix=mMpiLocal.mpiDistributeMatrix(b,0);
    // MatrixUtilities::debugMatrixDifferentCpus(cpuRank,meshRowSize,meshColumnSize,aLocalMatrix,"");
    // double* matrixARecovered=MatrixUtilities::matrixMemoryAllocation(rowsA,columnsA);
    // matrixARecovered=mMpiLocal.mpiRecoverDistributedMatrixGatherV(aLocalMatrix,0);
    // if(cpuRank==0)
    // {
    //     usleep(2000);
    //     cout<<"Matrix recuperada: "<<endl;
    //     MatrixUtilities::printMatrix(rowsA,columnsA,matrixARecovered);
    // }
    double *cLocalMatrix= MatrixUtilities::matrixMemoryAllocation(mMpiLocal.getBlockRowSize(),mMpiLocal.getBlockColumnSize());
    cLocalMatrix=mMpiLocal.mpiSumma(rowsA,rowsA,rowsA,aLocalMatrix,bLocalMatrix,meshRowColumnSize,meshRowColumnSize);
    // MatrixUtilities::debugMatrixDifferentCpus(cpuRank,meshRowSize,meshColumnSize,cLocalMatrix,"");
    double* matrixFinalRes=MatrixUtilities::matrixMemoryAllocation(rowsA,columnsB);
    matrixFinalRes=mMpiLocal.mpiRecoverDistributedMatrixReduce(cLocalMatrix,0);
    MatrixUtilities::printMatrixOrMessageForOneCpu(rowsA,columnsB,matrixFinalRes,cpuRank,0,"El resultado de la multiplicacion es: ");
    MPI_Finalize();
}
