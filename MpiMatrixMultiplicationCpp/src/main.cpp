#include <string>
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <iomanip>
#include <math.h>
#include "anyoption.h"
#include "MatrixMain.h"
#include "MpiMultiplicationEnvironment.h"
#include "MatrixUtilities.h"
#include "OperationProperties.h"

using namespace std;

template <class Toperation>
Toperation *PerformCalculations(MatrixMain<Toperation> *ma, MatrixMain<Toperation> *mb, OperationProperties op, int root, MPI_Comm commOperation, bool printMatrix)
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
    if (cpuRank == root)
    {
        int rowsAReal = ma->getRowsReal();
        int columnsBUsed = mb->getColumnsUsed();
        int columnsBReal = mb->getColumnsReal();
        Toperation *matrixWithout0 = MatrixUtilities<Toperation>::getMatrixWithoutZeros(rowsAReal, columnsBUsed, columnsBReal, matrixFinalRes);
        MatrixUtilities<Toperation>::printMatrixOrMessageForOneCpu(rowsA, columnsB, matrixFinalRes, cpuRank, root, "Dimensiones C: Rows" + to_string(rowsA) + ", Columns: " + to_string(columnsB) + ", El resultado de la multiplicacion es: ");

        MatrixUtilities<Toperation>::printMatrixOrMessageForOneCpu(rowsAReal, columnsBReal, matrixWithout0, cpuRank, root, "Dimensiones C: Rows" + to_string(rowsAReal) + ", Columns: " + to_string(columnsBReal) + ", Sin los 0s: ");
        return matrixWithout0;
    }
}

int main(int argc, char *argv[])
{
    int cpuRank, cpuSize, root, cpuOperationsSize,i;
    double timeDistributedOperationInitial, timeDistributedOperationFinal;
    bool printMatrix=false;
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
        for(i =0;i<argc;i++)
        {
            optionsCmd.push_back(string(argv[i]));
            cout<<optionsCmd[i]<<endl;
        }
        if (std::find(optionsCmd.begin(), optionsCmd.end(),"-h")!=optionsCmd.end())
        {
            cout<<"Hola"<<endl;
        }
        // }
        // AnyOption *opt = new AnyOption();
        // opt->addUsage("Uso: ");
        // opt->addUsage("Las opciones -f y -r no se pueden usar a la vez");
        // opt->addUsage(" -h          	Muestra la ayuda ");
        // opt->addUsage(" -p   			(Opcional) Muestra las matrices por pantalla ");
        // opt->addUsage(" -f   			Las matrices son leidas de ficheros de texto: -f f1.txt f2.txt ");
        // opt->addUsage(" -r   			Las matrices son generadas de forma aleatoria(m n k indican el tamaño de las matrices. bl bu indican de que numero a que numero se genera la matrix .Todos son numeros enteros) -r m n k ");
        // // opt->processCommandArgs(10, argv);
        // if (!opt->hasOptions())
        // {
        //     cout << "No se han proporcionado los parametros adecuados de uso. La forma de uso es la siguiente: " << endl;
        //     opt->printUsage();
        //     delete opt;
        //     MPI_Abort(MPI_COMM_WORLD, -1);
        //     return -1;
        // }
        // if (opt->getFlag("h"))
        // {
        //     opt->printUsage();
        // }
        // if (opt->getFlag("p"))
        // {
        //     printMatrix=true;
        // }
        // if (opt->getFlag("r"))
        // {
        //     ma = new MatrixMain<double>(atoi(opt->getArgv(0)),atoi(opt->getArgv(1)),atoi(opt->getArgv(3)),atoi(opt->getArgv(4)));
        //     mb = new MatrixMain<double>(atoi(opt->getArgv(1)),atoi(opt->getArgv(2)),atoi(opt->getArgv(3)),atoi(opt->getArgv(4)));
        // }
        // if (opt->getFlag("f"))
        // {
        //     cout<<"HOLA"<<endl;
        //     ma = new MatrixMain<double>(opt->getArgv(0));
        //     mb = new MatrixMain<double>(opt->getArgv(1));
        // }
        // delete opt;
        if (!MatrixUtilities<double>::canMultiply(ma->getColumnsReal(), mb->getRowsReal()))
        {
            //ABORTAMOS porque no cumple la regla de multiplicacion de matrices
            cout << "Las dimensiones de A:" << endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
            return -1;
        }
        op = MatrixUtilities<double>::getMeshAndMatrixSize(ma->getRowsReal(), ma->getColumnsReal(), mb->getRowsReal(), mb->getColumnsReal(), cpuSize);
        cpuOperationsSize = op.cpuSize;
    }
    //El calculo solo lo haran los procesadores seleccionados
    MPI_Bcast(&cpuOperationsSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
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
        distributedRes = PerformCalculations(ma, mb, op, root, commOperation,printMatrix);
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
        auto errors = MatrixUtilities<double>::checkEqualityOfMatrices(res, distributedRes, ma->getRowsReal(), mb->getColumnsReal());
        MatrixUtilities<double>::printErrorEqualityMatricesPosition(errors);
        MatrixUtilities<double>::matrixFree(matrixWithout0A);
        MatrixUtilities<double>::matrixFree(matrixWithout0B);
        MatrixUtilities<double>::matrixFree(res);
        delete ma,mb;

    }

    MPI_Finalize();
}