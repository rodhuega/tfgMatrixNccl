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

template <class Toperation>
Toperation* PerformCalculations(MatrixMain<Toperation> *ma, MatrixMain<Toperation> *mb, OperationProperties op, int root, MPI_Comm commOperation)
{
    int cpuRank, cpuSize;
    int rowsA, columnsA, rowsB, columnsB, meshRowSize, meshColumnSize;
    Toperation *a = NULL;
    Toperation *b = NULL;
    MPI_Comm_size(commOperation, &cpuSize);
    MPI_Comm_rank(commOperation, &cpuRank);

    if (cpuRank == root)
    {
        meshRowSize = op.meshRowSize;
        meshColumnSize = op.meshColumnSize;
        cout << "meshRowSize: " << meshRowSize << ", meshColumnSize: " << meshColumnSize << endl;
        ma->setRowsUsed(op.rowsA);
        ma->setColumnsUsed(op.columnsAorRowsB);
        ma->fillMatrix(false);
        a = ma->getMatrix();
        rowsA = ma->getRowsUsed();
        columnsA = ma->getColumnsUsed();

        mb->setRowsUsed(op.columnsAorRowsB);
        mb->setColumnsUsed(op.columnsB);
        mb->fillMatrix(false);
        b = mb->getMatrix();
        rowsB = mb->getRowsUsed();
        columnsB = mb->getColumnsUsed();

        cout << "A-> Rows: " << rowsA << ", Columns: " << columnsA << ", Matriz A:" << endl;
        MatrixUtilities<Toperation>::printMatrix(rowsA, columnsA, a);
        cout << "B-> Rows: " << rowsB << ", Columns: " << columnsB << ", Matriz B:" << endl;
        MatrixUtilities<Toperation>::printMatrix(rowsB, columnsB, b);
    }
    //Broadcasting de informacion basica pero necesaria
    MPI_Bcast(&rowsA, 1, MPI_INT, 0, commOperation);
    MPI_Bcast(&columnsA, 1, MPI_INT, 0, commOperation);
    MPI_Bcast(&rowsB, 1, MPI_INT, 0, commOperation);
    MPI_Bcast(&columnsB, 1, MPI_INT, 0, commOperation);
    MPI_Bcast(&meshRowSize, 1, MPI_INT, 0, commOperation);
    MPI_Bcast(&meshColumnSize, 1, MPI_INT, 0, commOperation);

    //Distribucion de las matrices entre los distintos procesos
    MpiMatrix<Toperation> mMpiLocalA = MpiMatrix<Toperation>(cpuSize, cpuRank, meshRowSize, meshColumnSize, rowsA, columnsA, commOperation);
    MpiMatrix<Toperation> mMpiLocalB = MpiMatrix<Toperation>(cpuSize, cpuRank, meshRowSize, meshColumnSize, rowsB, columnsB, commOperation);

    // cout << "Procedemos a distribuir A:" << endl;
    mMpiLocalA.mpiDistributeMatrix(a, 0);
    // MatrixUtilities::debugMatrixDifferentCpus(cpuRank,mMpiLocalA.getBlockRowSize(),mMpiLocalA.getBlockColumnSize(),mMpiLocalA.getMatrixLocal(),"");

    // cout << "Procedemos a distribuir B:" << endl;

    mMpiLocalB.mpiDistributeMatrix(b, 0);
    // MatrixUtilities::debugMatrixDifferentCpus(cpuRank,mMpiLocalB.getBlockRowSize(),mMpiLocalB.getBlockColumnSize(),mMpiLocalB.getMatrixLocal(),"");
    // usleep(10000);
    // double* matrixARecovered=MatrixUtilities::matrixMemoryAllocation(rowsA,columnsA);
    // matrixARecovered=mMpiLocalA.mpiRecoverDistributedMatrixGatherV(root);
    // double* matrixBRecovered=MatrixUtilities::matrixMemoryAllocation(rowsB,columnsB);
    // matrixBRecovered=mMpiLocalB.mpiRecoverDistributedMatrixGatherV(root);
    // if(cpuRank==root)
    // {
    //     usleep(2000);
    //     cout<<"Matrix recuperada: "<<endl;
    //     MatrixUtilities::printMatrix(rowsA,columnsA,matrixARecovered);
    // }
    MpiMultiplicationEnvironment<Toperation> mpiMultEnv = MpiMultiplicationEnvironment<Toperation>(cpuRank, cpuSize, commOperation);

    MpiMatrix<Toperation> mMpiLocalC = mpiMultEnv.mpiSumma(mMpiLocalA, mMpiLocalB, meshRowSize, meshColumnSize);
    // // MatrixUtilities::debugMatrixDifferentCpus(cpuRank,meshRowSize,meshColumnSize,mMpiLocalC.getMatrixLocal(),"");
    double *matrixFinalRes = mMpiLocalC.mpiRecoverDistributedMatrixReduce(root);
    MatrixUtilities<Toperation>::printMatrixOrMessageForOneCpu(rowsA, columnsB, matrixFinalRes, cpuRank, root, "Dimensiones C: Rows" + to_string(rowsA) + ", Columns: " + to_string(columnsB) + ", El resultado de la multiplicacion es: ");
    if (cpuRank == root)
    {
        int rowsAReal = ma->getRowsReal();
        int columnsBUsed = mb->getColumnsUsed();
        int columnsBReal = mb->getColumnsReal();
        Toperation *matrixWithout0 = MatrixUtilities<Toperation>::getMatrixWithoutZeros(rowsAReal, columnsBUsed, columnsBReal, matrixFinalRes);
        MatrixUtilities<Toperation>::printMatrixOrMessageForOneCpu(rowsAReal, columnsBReal, matrixWithout0, cpuRank, root, "Dimensiones C: Rows" + to_string(rowsAReal) + ", Columns: " + to_string(columnsBReal) + ", Sin los 0s: ");
        return matrixWithout0;
    }
}

int main(int argc, char *argv[])
{
    int cpuRank, cpuSize, root, cpuOperationsSize, i;
    double timeDistributedOperationInitial, timeDistributedOperationFinal;
    double * distributedRes;
    MPI_Group groupInitial;
    MPI_Comm commOperation;
    MPI_Group groupOperation;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &cpuSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &cpuRank);
    root = 0;
    cout << fixed;
    cout << setprecision(2);
    MatrixMain<double> *ma;
    MatrixMain<double> *mb;
    OperationProperties op;
    //Acciones iniciales solo realizadas por la cpu root
    if (cpuRank == root)
    {
        ma = new MatrixMain<double>(argv[1]);
        mb = new MatrixMain<double>(argv[2]);
        if (!MatrixUtilities<double>::canMultiply(ma->getColumnsReal(), mb->getRowsReal()))
        {
            //ABORTAMOS porque no cumple la regla de multiplicacion de matrices
            cout << "Las dimensiones de A:" << endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        op = MatrixUtilities<double>::getMeshAndMatrixSize(ma->getRowsReal(), ma->getColumnsReal(), mb->getRowsReal(), mb->getColumnsReal(), cpuSize);
        cpuOperationsSize = op.cpuSize;
    }
    //El calculo solo lo haran los procesadores seleccionados
    MPI_Bcast(&cpuOperationsSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int membersGroupOperation[cpuOperationsSize];
    for (i = 0; i < cpuOperationsSize; i++)
    {
        membersGroupOperation[i] = i;
    }
    MPI_Comm_group(MPI_COMM_WORLD, &groupInitial);
    MPI_Group_incl(groupInitial, cpuOperationsSize, membersGroupOperation, &groupOperation);
    MPI_Comm_create(MPI_COMM_WORLD, groupOperation, &commOperation);
    if (cpuRank == root)
    {
        timeDistributedOperationFinal = MPI_Wtime();
    }
    if (cpuRank < cpuOperationsSize)
    {
        distributedRes=PerformCalculations(ma, mb, op, root, commOperation);
    }
    if (cpuRank == root)
    {
        timeDistributedOperationFinal = MPI_Wtime();
        double tTotal = timeDistributedOperationFinal - timeDistributedOperationInitial;
        cout << "El tiempo de calculo de la matriz de forma distribuida ha sido de: " << tTotal << endl;
        cout << "Calculando la matriz de forma no distribuida" << endl;
        double *matrixWithout0A = MatrixUtilities<double>::getMatrixWithoutZeros(ma->getRowsReal(), ma->getColumnsUsed(), ma->getColumnsReal(), ma->getMatrix());
        double *matrixWithout0B = MatrixUtilities<double>::getMatrixWithoutZeros(mb->getRowsReal(), mb->getColumnsUsed(), mb->getColumnsReal(), mb->getMatrix());
        double *res = MatrixUtilities<double>::matrixMemoryAllocation(ma->getRowsReal(), mb->getColumnsReal());
        MatrixUtilities<double>::Multiplicacion(ma->getRowsReal(), ma->getColumnsReal(), mb->getColumnsReal(), matrixWithout0A, matrixWithout0B, res);
        MatrixUtilities<double>::printMatrixOrMessageForOneCpu(ma->getRowsReal(), mb->getColumnsReal(), res, cpuRank, root, "Resultado sin distribuir: ");
        auto errors=MatrixUtilities<double>::checkEqualityOfMatrices(res,distributedRes,ma->getRowsReal(),mb->getColumnsReal());
        MatrixUtilities<double>::printErrorEqualityMatricesPosition(errors);
    }

    MPI_Finalize();
}