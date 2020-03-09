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
 * @brief Metodo que realiza el calculo de la matriz de forma secuencial y las compara. Ademas devuelve el tiempo que ha tardado el calculo secuencial
 * 
 * @tparam Toperation , tipo de la matriz(double,int,float)
 * @param ma , Matriz global A
 * @param mb , Matriz global B
 * @param distributedRes , Matriz resultado global de la operacion distribuida
 * @param root , id de la cpu que actuara como root
 * @param cpuRank , id de la cpu actual
 * @param printMatrix , indica si se desea que se impriman las matrices
 * @return double
 */
template <class Toperation>
double finalInstructionsForRoot(Toperation *ma, Toperation *mb, Toperation *distributedRes, int root, int cpuRank, bool printMatrix,int rowsA,int columnsAorRowsB, int columnsB)
{
    //Calculo secuencial
    cout << "Calculando la matriz de forma no distribuida" << endl;
    Toperation *res = MatrixUtilities<Toperation>::matrixMemoryAllocation(rowsA, columnsB);
    double tIniSingleCpu = MPI_Wtime();
    MatrixUtilities<Toperation>::matrixBlasMultiplication(rowsA, columnsAorRowsB, columnsB, ma, mb, res);
    double tFinSingleCpu = MPI_Wtime();
    double tTotalSingleCpu = tFinSingleCpu - tIniSingleCpu;
    cout << "El tiempo de calculo de la matriz de forma secuencial ha sido de: " << tTotalSingleCpu << endl;
    cout << setprecision(2);
    if (printMatrix)
    {
        MatrixUtilities<Toperation>::printMatrixOrMessageForOneCpu(rowsA, columnsB, res, cpuRank, root, "Resultado sin distribuir: ");
    }
    //Commparacion
    auto errors = MatrixUtilities<Toperation>::checkEqualityOfMatrices(res, distributedRes, rowsA, columnsB);
    MatrixUtilities<Toperation>::printErrorEqualityMatricesPosition(errors,false);
    MatrixUtilities<Toperation>::matrixFree(res);
    return tTotalSingleCpu;
}

int main(int argc, char *argv[])
{
    //////////////////////////////SE MARCARA CON //** las instrucciones necesarias para operar con la libreria y con //*** las opcionales
    int cpuRank, cpuSize, root, cpuOperationsSize, i;
    double timeDistributedOperationInitial, timeDistributedOperationFinal,tTotal;
    bool printMatrix = false;
    int rowsA,columnsAorRowsB,columnsB;
    //Si se quiere cambiar el tipo de las operaciones hay que cambiar por ejemplo los <double> a <float> y los tipos que hay entre estos dos comentarios
    double *distributedRes;
    MPI_Datatype basicOperationType = MPI_DOUBLE;
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
    double *matrixA=nullptr;
    double *matrixB=nullptr;
    MpiMultiplicationEnvironment<double> mpiMult = MpiMultiplicationEnvironment<double>(cpuRank, root, cpuSize, basicOperationType);//**

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
            matrixA = MatrixUtilities<double>::ReadOrGenerateRandomMatrix(false, optionsCmd[fPosition + 1].c_str(), rowsA, columnsAorRowsB, -1, -1);
            matrixB = MatrixUtilities<double>::ReadOrGenerateRandomMatrix(false, optionsCmd[fPosition + 2].c_str(), columnsAorRowsB, columnsB, -1, -1);
        }

        if (rOptionChecker != optionsCmd.end())
        {
            int rPosition = std::distance(optionsCmd.begin(), rOptionChecker);
            rowsA=atoi(optionsCmd[rPosition + 1].c_str());
            columnsAorRowsB=atoi(optionsCmd[rPosition + 2].c_str());
            columnsB=atoi(optionsCmd[rPosition + 3].c_str());
            matrixA = MatrixUtilities<double>::ReadOrGenerateRandomMatrix(true, "", rowsA, columnsAorRowsB, atoi(optionsCmd[rPosition + 4].c_str()), atoi(optionsCmd[rPosition + 5].c_str()));
            matrixB = MatrixUtilities<double>::ReadOrGenerateRandomMatrix(true, "", columnsAorRowsB, columnsB, atoi(optionsCmd[rPosition + 4].c_str()), atoi(optionsCmd[rPosition + 5].c_str()));
        }
    }
    MPI_Bcast(&rowsA, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Bcast(&columnsAorRowsB, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Bcast(&columnsB, 1, MPI_INT, root, MPI_COMM_WORLD);
    mpiMult.setOrAddMatrixGlobalSimplePointer("A", matrixA,rowsA,columnsAorRowsB);//**
    mpiMult.setOrAddMatrixGlobalSimplePointer("B", matrixB,columnsAorRowsB,columnsB);//**
    if (cpuRank == root)
    {
        timeDistributedOperationInitial = MPI_Wtime();
    }
    
    mpiMult.PerformCalculations("A", "B", "C", printMatrix);//**
    //Comprobar si el calculo esta bien hecho y mostrar tiempos y liberacion de memoria solo por parte de la cpu root
    if (cpuRank == root)
    {
        timeDistributedOperationFinal = MPI_Wtime();
        tTotal = timeDistributedOperationFinal - timeDistributedOperationInitial;
    }
    
    //Recuperacion de la matriz resultado que estaba distribuida
    if(mpiMult.getIfThisCpuPerformOperation())
    {
        mpiMult.recoverDistributedMatrix("C");//***
        distributedRes=mpiMult.getMatrixGlobalSimplePointer("C");//***
    }
    if(cpuRank==root)
    {
        //Mostrar informacion en caso de que sea necesario
        if (printMatrix)
        {
            MatrixUtilities<double>::printMatrixOrMessageForOneCpu(rowsA, columnsB, distributedRes, cpuRank, root, "Dimensiones C: Rows: " + std::to_string(rowsA) + ", Columns: " + std::to_string(columnsB) + ", El resultado de la multiplicacion es: ");
        }
        cout << setprecision(6);
        cout << "El tiempo de calculo de la matriz de forma distribuida ha sido de: " << tTotal << endl;
        double singleCPU=finalInstructionsForRoot<double>(matrixA,matrixB, distributedRes, root, cpuRank, printMatrix,rowsA,columnsAorRowsB,columnsB);
        //Calculo de la aceleracion
        double speedUp=singleCPU/tTotal;
        cout << setprecision(6);
        cout<<"La aceleracion ha sido: "<<speedUp<<endl;
        //Liberacion de recuersos
        MatrixUtilities<double>::matrixFree(distributedRes);
        MatrixUtilities<double>::matrixFree(matrixA);
        MatrixUtilities<double>::matrixFree(matrixB);
    }
    MPI_Finalize();
}