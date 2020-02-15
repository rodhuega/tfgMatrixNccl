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
    int cpuRank, cpuSize,root;
    int rowsA, columnsA, rowsB, columnsB,meshRowSize,meshColumnSize;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &cpuSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &cpuRank);
    root=0;
    cout<<fixed;
    cout<<setprecision(2);
    double *a = NULL;
    double *b = NULL;
    //Actiones iniciales solo realizadas por la cpu root
    if (cpuRank == root)
    {
        MatrixMain ma = MatrixMain(argv[1]);
        MatrixMain mb = MatrixMain(argv[2]);
        if(!MatrixUtilities::canMultiply(ma.getColumnsReal(),mb.getRowsReal()))
        {
            //ABORTAMOS porque no cumple la regla de multiplicacion de matrices
            cout<<"Las dimensiones de A:"<<endl;
            MPI_Abort(MPI_COMM_WORLD,-1);
        }
        int* operationProperties=MatrixUtilities::getMeshAndMatrixSize(ma.getRowsReal(),ma.getColumnsReal(),mb.getRowsReal(),mb.getColumnsReal(),cpuSize);
        meshRowSize=operationProperties[0];
        meshColumnSize=operationProperties[1];
        cout<<"meshRowSize: "<<meshRowSize<<", meshColumnSize: "<<meshColumnSize<<", operationProperties[2]:"<<operationProperties[3]<<endl;
        ma.setRowsUsed(operationProperties[2]);
        ma.setColumnsUsed(operationProperties[3]);
        ma.fillMatrix(false);
        a = ma.getMatrix();
        rowsA = ma.getRowsUsed();
        columnsA = ma.getColumnsUsed();

        mb.setRowsUsed(operationProperties[2]);
        mb.setColumnsUsed(operationProperties[3]);
        mb.fillMatrix(false);
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
    //Broadcasting de informacion basica pero necesaria
    MPI_Bcast(&rowsA, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&columnsA, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rowsB, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&columnsB, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&meshRowSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&meshColumnSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    //Distribucion de las matrices entre los distintos procesos
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
    cLocalMatrix=mMpiLocal.mpiSumma(rowsA,rowsA,rowsA,aLocalMatrix,bLocalMatrix,meshRowSize,meshColumnSize);
    // MatrixUtilities::debugMatrixDifferentCpus(cpuRank,meshRowSize,meshColumnSize,cLocalMatrix,"");
    double* matrixFinalRes=MatrixUtilities::matrixMemoryAllocation(rowsA,columnsB);
    matrixFinalRes=mMpiLocal.mpiRecoverDistributedMatrixReduce(cLocalMatrix,0);
    MatrixUtilities::printMatrixOrMessageForOneCpu(rowsA,columnsB,matrixFinalRes,cpuRank,0,"El resultado de la multiplicacion es: ");
    MPI_Finalize();
}
