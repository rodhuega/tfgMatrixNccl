#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <algorithm>
#include "MatrixUtilities.h"



using namespace std;

int main(int argc, char** argv) {
     //////////////////////////////SE MARCARA CON //** las instrucciones necesarias para operar con la libreria y con //*** las opcionales
    int gpuSize, gpuRoot, cpuOperationsSize, i;
    double timeDistributedOperationInitial, timeDistributedOperationFinal,tTotal;
    bool printMatrix = false;
    int rowsA,columnsA,rowsB,columnsB;
    //Si se quiere cambiar el tipo de las operaciones hay que cambiar por ejemplo los <double> a <float> y los tipos que hay entre estos dos comentarios
    double *distributedRes;
    gpuRoot = 0;
    cout << fixed;
    cout << setprecision(2);
    double *matrixA=nullptr;
    double *matrixB=nullptr;
    // MpiMultiplicationEnvironment<double> mpiMult = MpiMultiplicationEnvironment<double>(cpuRank, root, cpuSize, basicOperationType);//**
    
    //Lectura de los parametros de lanzamiento
    vector<string> optionsCmd;
    for (i = 0; i < argc; i++)
    {
        optionsCmd.push_back(string(argv[i]));
    }
    auto fOptionChecker = std::find(optionsCmd.begin(), optionsCmd.end(), "-f");
    auto rOptionChecker = std::find(optionsCmd.begin(), optionsCmd.end(), "-r");
    auto gOptionChecker = std::find(optionsCmd.begin(), optionsCmd.end(), "-g");
    if (std::find(optionsCmd.begin(), optionsCmd.end(), "-h") != optionsCmd.end() || optionsCmd.size() == 0 ||(fOptionChecker == optionsCmd.end() && rOptionChecker == optionsCmd.end()))
    {
        cout << "Uso:\tLas opciones -f y -r no se pueden usar a la vez" << endl;
        cout << "\t-h\tMuestra la ayuda" << endl;
        cout << "\t-p\t(Opcional) Muestra las matrices por pantalla" << endl;
        cout << "\t-f\tLas matrices son leidas de ficheros de texto: -f f1.txt f2.txt" << endl;
        cout << "\t-r\tLas matrices son generadas de forma aleatoria(m n k indican el tamaño de las matrices. bl bu indican de que numero a que numero se genera la matrix .Todos son numeros enteros) -r m n k " << endl;
        cout << "\t-g\t(Opcional) Indica el número de gpus que van a ser usadas. Pueden ser menos de las que hay en el sistema o más y se usaran algunas varias veces para simular ese número. Ejemplo -g 4" << endl;
    }
    if (std::find(optionsCmd.begin(), optionsCmd.end(), "-p") != optionsCmd.end())
    {
        printMatrix = true;
    }
    if (gOptionChecker != optionsCmd.end())
    {
        int gPosition = std::distance(optionsCmd.begin(), gOptionChecker);
        gpuSize = atoi(optionsCmd[gPosition + 1].c_str());
    }
    
    if (fOptionChecker != optionsCmd.end() && rOptionChecker != optionsCmd.end())
    {
        cout << "Los parametros -f y -r no se pueden usar a la vez" << endl;
        return -1;
    }
    if (fOptionChecker != optionsCmd.end())
    {
        int fPosition = std::distance(optionsCmd.begin(), fOptionChecker);
        matrixA = MatrixUtilities<double>::ReadOrGenerateRandomMatrix(false, optionsCmd[fPosition + 1].c_str(), rowsA, columnsA, -1, -1);
        matrixB = MatrixUtilities<double>::ReadOrGenerateRandomMatrix(false, optionsCmd[fPosition + 2].c_str(), rowsB, columnsB, -1, -1);
    }

    if (rOptionChecker != optionsCmd.end())
    {
        int rPosition = std::distance(optionsCmd.begin(), rOptionChecker);
        rowsA=atoi(optionsCmd[rPosition + 1].c_str());
        columnsA=atoi(optionsCmd[rPosition + 2].c_str());
        rowsB=atoi(optionsCmd[rPosition + 2].c_str());
        columnsB=atoi(optionsCmd[rPosition + 3].c_str());
        matrixA = MatrixUtilities<double>::ReadOrGenerateRandomMatrix(true, "", rowsA, columnsA, atoi(optionsCmd[rPosition + 4].c_str()), atoi(optionsCmd[rPosition + 5].c_str()));
        matrixB = MatrixUtilities<double>::ReadOrGenerateRandomMatrix(true, "", rowsB, columnsB, atoi(optionsCmd[rPosition + 4].c_str()), atoi(optionsCmd[rPosition + 5].c_str()));
    }
    MatrixUtilities<double>::printMatrix(rowsA,columnsA,matrixA);
}
