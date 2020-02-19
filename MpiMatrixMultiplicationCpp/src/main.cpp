#include <string>
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <iomanip>
#include <math.h>
#include <algorithm>
#include "MatrixMain.h"
#include "MpiMultiplicationEnvironment.h"
#include "MatrixUtilities.h"
#include "OperationProperties.h"

using namespace std;

template <class Toperation>
Toperation *PerformCalculations(MatrixMain<Toperation> *ma, MatrixMain<Toperation> *mb, OperationProperties op, int root, MPI_Comm commOperation, bool printMatrix,bool isRandom)
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
        ma->fillMatrix(isRandom);
        a = ma->getMatrix();
        rowsA = ma->getRowsUsed();
        columnsA = ma->getColumnsUsed();

        mb->setRowsUsed(op.columnsAorRowsB);
        mb->setColumnsUsed(op.columnsB);
        mb->fillMatrix(isRandom);
        b = mb->getMatrix();
        rowsB = mb->getRowsUsed();
        columnsB = mb->getColumnsUsed();
        if (printMatrix)
        {
            cout << "A-> Rows: " << rowsA << ", Columns: " << columnsA << ", Matriz A:" << endl;
            MatrixUtilities<Toperation>::printMatrix(rowsA, columnsA, a);
            cout << "B-> Rows: " << rowsB << ", Columns: " << columnsB << ", Matriz B:" << endl;
            MatrixUtilities<Toperation>::printMatrix(rowsB, columnsB, b);
        }
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
    mMpiLocalA.mpiDistributeMatrix(a, root);
    // MatrixUtilities<Toperation>::debugMatrixDifferentCpus(cpuRank,mMpiLocalA.getBlockRowSize(),mMpiLocalA.getBlockColumnSize(),mMpiLocalA.getMatrixLocal(),"");

    // cout << "Procedemos a distribuir B:" << endl;

    mMpiLocalB.mpiDistributeMatrix(b, root);
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
    if (cpuRank == root)
    {
        int rowsAReal = ma->getRowsReal();
        int columnsBUsed = mb->getColumnsUsed();
        int columnsBReal = mb->getColumnsReal();
        Toperation *matrixWithout0 = MatrixUtilities<Toperation>::getMatrixWithoutZeros(rowsAReal, columnsBUsed, columnsBReal, matrixFinalRes);
        if (printMatrix)
        {
            MatrixUtilities<Toperation>::printMatrixOrMessageForOneCpu(rowsA, columnsB, matrixFinalRes, cpuRank, root, "Dimensiones C: Rows: " + to_string(rowsA) + ", Columns: " + to_string(columnsB) + ", El resultado de la multiplicacion es: ");
            MatrixUtilities<Toperation>::printMatrixOrMessageForOneCpu(rowsAReal, columnsBReal, matrixWithout0, cpuRank, root, "Dimensiones C: Rows: " + to_string(rowsAReal) + ", Columns: " + to_string(columnsBReal) + ", Sin los 0s: ");
        }
        return matrixWithout0;
    }
}

int main(int argc, char *argv[])
{
    int cpuRank, cpuSize, root, cpuOperationsSize, i;
    double timeDistributedOperationInitial, timeDistributedOperationFinal;
    bool printMatrix = false;
    bool isRandom=false;
    double *distributedRes;
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
        vector<string> optionsCmd;
        for (i = 0; i < argc; i++)
        {
            optionsCmd.push_back(string(argv[i]));
        }
        if (std::find(optionsCmd.begin(), optionsCmd.end(), "-h") != optionsCmd.end() || optionsCmd.size() == 0)
        {
            cout << "Uso:\tLas opciones -f y -r no se pueden usar a la vez" << endl;
            cout << "\t-h\tMuestra la ayuda" << endl;
            cout << "\t-p\t(Opcional) Muestra las matrices por pantalla" << endl;
            cout << "\t-f\tLas matrices son leidas de ficheros de texto: -f f1.txt f2.txt" << endl;
            cout << "\t-r\tLas matrices son generadas de forma aleatoria(m n k indican el tamaño de las matrices. bl bu indican de que numero a que numero se genera la matrix .Todos son numeros enteros) -r m n k " << endl;
        }
        if (std::find(optionsCmd.begin(), optionsCmd.end(), "-p") != optionsCmd.end())
        {
            printMatrix = true;
        }
        auto fOptionChecker = std::find(optionsCmd.begin(), optionsCmd.end(), "-f");
        auto rOptionChecker = std::find(optionsCmd.begin(), optionsCmd.end(), "-r");
        if (fOptionChecker != optionsCmd.end() && rOptionChecker != optionsCmd.end())
        {
            cout << "Los parametros -f y -r no se pueden usar a la vez" << endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
            return -1;
        }
        if (fOptionChecker != optionsCmd.end())
        {
            int fPosition = std::distance(optionsCmd.begin(), fOptionChecker);
            ma = new MatrixMain<double>(optionsCmd[fPosition + 1].c_str());
            mb = new MatrixMain<double>(optionsCmd[fPosition + 2].c_str());
        }

        if (rOptionChecker != optionsCmd.end())
        {
            isRandom=true;
            int rPosition = std::distance(optionsCmd.begin(), rOptionChecker);
            ma = new MatrixMain<double>(atoi(optionsCmd[rPosition + 1].c_str()), atoi(optionsCmd[rPosition + 2].c_str()), atoi(optionsCmd[rPosition + 4].c_str()), atoi(optionsCmd[rPosition + 5].c_str()));
            mb = new MatrixMain<double>(atoi(optionsCmd[rPosition + 2].c_str()), atoi(optionsCmd[rPosition + 3].c_str()), atoi(optionsCmd[rPosition + 4].c_str()), atoi(optionsCmd[rPosition + 5].c_str()));
        }

        if (!MatrixUtilities<double>::canMultiply(ma->getColumnsReal(), mb->getRowsReal()))
        {
            //ABORTAMOS porque no cumple la regla de multiplicacion de matrices
            cout << "Las matrices no se pueden multiplicar entre si" << endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
            return -1;
        }
        op = MatrixUtilities<double>::getMeshAndMatrixSize(ma->getRowsReal(), ma->getColumnsReal(), mb->getRowsReal(), mb->getColumnsReal(), cpuSize);
        cpuOperationsSize = op.cpuSize;
    }
    //El calculo solo lo haran los procesadores seleccionados
    MPI_Bcast(&cpuOperationsSize, 1, MPI_INT, root, MPI_COMM_WORLD);
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
        cout<<"Por aqui todo bien"<<endl;
        distributedRes = PerformCalculations(ma, mb, op, root, commOperation, printMatrix,isRandom);
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
        double tIniSingleCpu = MPI_Wtime();
        MatrixUtilities<double>::Multiplicacion(ma->getRowsReal(), ma->getColumnsReal(), mb->getColumnsReal(), matrixWithout0A, matrixWithout0B, res);
        double tFinSingleCpu = MPI_Wtime();
        double tTotalSingleCpu = tFinSingleCpu - tIniSingleCpu;
        cout << "El tiempo de calculo de la matriz de forma secuencial ha sido de: " << tTotalSingleCpu << endl;
        if (printMatrix)
        {
            MatrixUtilities<double>::printMatrixOrMessageForOneCpu(ma->getRowsReal(), mb->getColumnsReal(), res, cpuRank, root, "Resultado sin distribuir: ");
        }
        auto errors = MatrixUtilities<double>::checkEqualityOfMatrices(res, distributedRes, ma->getRowsReal(), mb->getColumnsReal());
        MatrixUtilities<double>::printErrorEqualityMatricesPosition(errors);
        MatrixUtilities<double>::matrixFree(matrixWithout0A);
        MatrixUtilities<double>::matrixFree(matrixWithout0B);
        MatrixUtilities<double>::matrixFree(res);
        delete ma, mb;
    }

    MPI_Finalize();
}