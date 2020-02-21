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

/**
 * @brief Metodo que se encarga de realizar el calculo de la multiplicacion distribuida
 * 
 * @tparam Toperation , tipo de la matriz(double,int,float)
 * @param ma , Matriz global A
 * @param mb , Matriz global B
 * @param op , propiedades de la operacion(cpus,tamaños de mallas)
 * @param root , id de la cpu que actuara como root
 * @param commOperation , Comunidacdor mpi para los procesos que van a realizar el calculo
 * @param printMatrix , indica si se desea que se impriman las matrices
 * @param isRandom , indica si se van a generar matrices A y B de forma aleatoria
 * @param basicOperationType tipo mpi basico del que se van a realizar las operaciones(MPI_DOUBLE,MPI_INT). Debe de coincidir con Toperation
 * @return Toperation* , matriz global resultado de la multiplicacion de forma distribuida
 */
template <class Toperation>
Toperation *PerformCalculations(MatrixMain<Toperation> *ma, MatrixMain<Toperation> *mb, OperationProperties op, int root, MPI_Comm commOperation, bool printMatrix, bool isRandom, MPI_Datatype basicOperationType)
{
    int cpuRank, cpuSize;
    int rowsA, columnsA, rowsB, columnsB, meshRowSize, meshColumnSize;
    Toperation *a = NULL;
    Toperation *b = NULL;
    MPI_Comm_size(commOperation, &cpuSize);
    MPI_Comm_rank(commOperation, &cpuRank);
    //Creacion de las matrices por parte del proceso root
    if (cpuRank == root)
    {
        meshRowSize = op.meshRowSize;
        meshColumnSize = op.meshColumnSize;
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
    MpiMatrix<Toperation> mMpiLocalA = MpiMatrix<Toperation>(cpuSize, cpuRank, meshRowSize, meshColumnSize, rowsA, columnsA, commOperation, basicOperationType);
    MpiMatrix<Toperation> mMpiLocalB = MpiMatrix<Toperation>(cpuSize, cpuRank, meshRowSize, meshColumnSize, rowsB, columnsB, commOperation, basicOperationType);
    mMpiLocalA.mpiDistributeMatrix(a, root);
    mMpiLocalB.mpiDistributeMatrix(b, root);
    //Realizacion de la multiplicacion distribuicda
    MpiMultiplicationEnvironment<Toperation> mpiMultEnv = MpiMultiplicationEnvironment<Toperation>(cpuRank, cpuSize, commOperation, basicOperationType);
    MpiMatrix<Toperation> mMpiLocalC = mpiMultEnv.mpiSumma(mMpiLocalA, mMpiLocalB, meshRowSize, meshColumnSize);
    Toperation *matrixFinalRes = mMpiLocalC.mpiRecoverDistributedMatrixReduce(root);
    //Mostrar resultados y tiempos por parte de la cpu root
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
        MatrixUtilities<Toperation>::matrixFree(matrixFinalRes);
        return matrixWithout0;
    }else 
    {
        return NULL;
    }
}

/**
 * @brief Metodo que realiza el calculo de la matriz de forma secuencial y las compara
 * 
 * @tparam Toperation , tipo de la matriz(double,int,float)
 * @param ma , Matriz global A
 * @param mb , Matriz global B
 * @param distributedRes , Matriz resultado global de la operacion distribuida
 * @param root , id de la cpu que actuara como root
 * @param cpuRank , id de la cpu actual
 * @param printMatrix , indica si se desea que se impriman las matrices
 */
template <class Toperation>
void finalInstructionsForRoot(MatrixMain<Toperation> *ma, MatrixMain<Toperation> *mb, Toperation *distributedRes,int root,int cpuRank,bool printMatrix)
{
    cout << "Calculando la matriz de forma no distribuida" << endl;
    Toperation *matrixWithout0A = MatrixUtilities<Toperation>::getMatrixWithoutZeros(ma->getRowsReal(), ma->getColumnsUsed(), ma->getColumnsReal(), ma->getMatrix());
    Toperation *matrixWithout0B = MatrixUtilities<Toperation>::getMatrixWithoutZeros(mb->getRowsReal(), mb->getColumnsUsed(), mb->getColumnsReal(), mb->getMatrix());
    Toperation *res = MatrixUtilities<Toperation>::matrixMemoryAllocation(ma->getRowsReal(), mb->getColumnsReal());
    double tIniSingleCpu = MPI_Wtime();
    MatrixUtilities<Toperation>::Multiplicacion(ma->getRowsReal(), ma->getColumnsReal(), mb->getColumnsReal(), matrixWithout0A, matrixWithout0B, res);
    double tFinSingleCpu = MPI_Wtime();
    double tTotalSingleCpu = tFinSingleCpu - tIniSingleCpu;
    cout << "El tiempo de calculo de la matriz de forma secuencial ha sido de: " << tTotalSingleCpu << endl;
    if (printMatrix)
    {
        MatrixUtilities<Toperation>::printMatrixOrMessageForOneCpu(ma->getRowsReal(), mb->getColumnsReal(), res, cpuRank, root, "Resultado sin distribuir: ");
    }
    auto errors = MatrixUtilities<Toperation>::checkEqualityOfMatrices(res, distributedRes, ma->getRowsReal(), mb->getColumnsReal());
    MatrixUtilities<Toperation>::printErrorEqualityMatricesPosition(errors);
    MatrixUtilities<Toperation>::matrixFree(matrixWithout0A);
    MatrixUtilities<Toperation>::matrixFree(matrixWithout0B);
    MatrixUtilities<Toperation>::matrixFree(res);
}

int main(int argc, char *argv[])
{
    int cpuRank, cpuSize, root, cpuOperationsSize, i;
    double timeDistributedOperationInitial, timeDistributedOperationFinal;
    bool printMatrix = false;
    bool isRandom = false;
    //Si se quiere cambiar el tipo de las operaciones hay que cambiar por ejemplo los <double> a <int> y los tipos que hay entre estos dos comentarios
    double *distributedRes;
    MPI_Datatype basicOperationType=MPI_DOUBLE;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
        //Lectura de los parametros de lanzamiento
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
            isRandom = true;
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
        //Optencion de los parametros necesarios para el calculo de la multiplicacion distribuida
        op = MatrixUtilities<double>::getMeshAndMatrixSize(ma->getRowsReal(), ma->getColumnsReal(), mb->getRowsReal(), mb->getColumnsReal(), cpuSize);
        cpuOperationsSize = op.cpuSize;
    }
    //El calculo solo lo haran los procesadores seleccionados, seleccionamos los procesadores que operaran
    MPI_Bcast(&cpuOperationsSize, 1, MPI_INT, root, MPI_COMM_WORLD);
    int membersGroupOperation[cpuOperationsSize];
    for (i = 0; i < cpuOperationsSize; i++)
    {
        membersGroupOperation[i] = i;
    }
    MPI_Comm_group(MPI_COMM_WORLD, &groupInitial);
    MPI_Group_incl(groupInitial, cpuOperationsSize, membersGroupOperation, &groupOperation);
    MPI_Comm_create(MPI_COMM_WORLD, groupOperation, &commOperation);
    //Calculo y medicion de tiempos
    if (cpuRank == root)
    {
        timeDistributedOperationInitial = MPI_Wtime();
    }
    if (cpuRank < cpuOperationsSize)
    {
        distributedRes = PerformCalculations(ma, mb, op, root, commOperation, printMatrix, isRandom, basicOperationType);
    }
    //Comprobar si el calculo esta bien hecho y mostrar tiempos y liberacion de memoria solo por parte de la cpu root
    if (cpuRank == root)
    {
        timeDistributedOperationFinal = MPI_Wtime();
        double tTotal = timeDistributedOperationFinal - timeDistributedOperationInitial;
        cout << "El tiempo de calculo de la matriz de forma distribuida ha sido de: " << tTotal << endl;
        finalInstructionsForRoot<double>(ma, mb,distributedRes,root,cpuRank,printMatrix);
        delete ma, mb;
    }
    MPI_Finalize();
}