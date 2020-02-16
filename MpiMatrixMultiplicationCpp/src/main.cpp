#include <string>
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <iomanip>
#include <math.h>
#include "MatrixMain.h"
#include "MpiMultiplicationEnvironment.h"
#include "MatrixUtilities.h"
#include "OperationProperties.h"

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
        OperationProperties op=MatrixUtilities::getMeshAndMatrixSize(ma.getRowsReal(),ma.getColumnsReal(),mb.getRowsReal(),mb.getColumnsReal(),cpuSize);
        meshRowSize=op.meshRowSize;
        meshColumnSize=op.meshColumnSize;
        cout<<"meshRowSize: "<<meshRowSize<<", meshColumnSize: "<<meshColumnSize<<endl;
        ma.setRowsUsed(op.rowsA);
        ma.setColumnsUsed(op.columnsAorRowsB);
        ma.fillMatrix(false);
        a = ma.getMatrix();
        rowsA = ma.getRowsUsed();
        columnsA = ma.getColumnsUsed();

        mb.setRowsUsed(op.columnsAorRowsB);
        mb.setColumnsUsed(op.columnsB);
        mb.fillMatrix(false);
        b = mb.getMatrix();
        rowsB = mb.getRowsUsed();
        columnsB = mb.getColumnsUsed();

        // cout << "La matriz A:" << endl;
        // MatrixUtilities::printMatrix(rowsA, columnsA, a);
        // cout << "La matriz B:" << endl;
        // MatrixUtilities::printMatrix(rowsB, columnsB, b);
    }
    //Broadcasting de informacion basica pero necesaria
    MPI_Bcast(&rowsA, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&columnsA, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rowsB, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&columnsB, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&meshRowSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&meshColumnSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    //Distribucion de las matrices entre los distintos procesos
    MpiMatrix mMpiLocalA= MpiMatrix(cpuSize,cpuRank,meshRowSize,meshColumnSize,rowsA,columnsA);
    MpiMatrix mMpiLocalB= MpiMatrix(cpuSize,cpuRank,meshRowSize,meshColumnSize,rowsB,columnsB);
    
    // cout << "Procedemos a distribuir A:" << endl;
    mMpiLocalA.mpiDistributeMatrix(a,0);
    // cout << "Procedemos a distribuir B:" << endl;
    mMpiLocalB.mpiDistributeMatrix(b,0);
    // MatrixUtilities::debugMatrixDifferentCpus(cpuRank,mMpiLocalB.getBlockRowSize(),mMpiLocalB.getBlockColumnSize(),mMpiLocalB.getMatrixLocal(),"");
    usleep(10000);
    // double* matrixARecovered=MatrixUtilities::matrixMemoryAllocation(rowsA,columnsA);
    // matrixARecovered=mMpiLocalA.mpiRecoverDistributedMatrixGatherV(root);
    // if(cpuRank==root)
    // {
    //     usleep(2000);
    //     cout<<"Matrix recuperada: "<<endl;
    //     MatrixUtilities::printMatrix(rowsA,columnsA,matrixARecovered);
    // }
    MpiMultiplicationEnvironment mpiMultEnv = MpiMultiplicationEnvironment(cpuRank,cpuSize);

    MpiMatrix mMpiLocalC=mpiMultEnv.mpiSumma(mMpiLocalA,mMpiLocalB,meshRowSize,meshColumnSize);
    // MatrixUtilities::debugMatrixDifferentCpus(cpuRank,meshRowSize,meshColumnSize,mMpiLocalC.getMatrixLocal(),"");
    double* matrixFinalRes=mMpiLocalC.mpiRecoverDistributedMatrixReduce(root);
    MatrixUtilities::printMatrixOrMessageForOneCpu(rowsA,columnsB,matrixFinalRes,cpuRank,root,"El resultado de la multiplicacion es: ");
    MPI_Finalize();
}
