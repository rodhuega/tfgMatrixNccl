#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <algorithm>

#include "MatrixUtilities.h"
#include "OperationType.h"

#include "MatrixUtilitiesCuda.cuh"
#include "NcclMultiplicationEnvironment.cuh"
#include "MatrixMain.cuh"

extern "C"
{
#include "ctimer.h"
}

using namespace std;

template <class Toperation>
void ejecucion(vector<string> optionsCmd, OperationType opt)
{
    int gpuSizeWorldArgument, cpuOperationsSize,gpuRoot=0;
    double elapsedDistributed, ucpuDistributed, scpuDistributed,elapsedGpuNoDistributed, ucpuGpuNoDistributed, scpuGpuNoDistributed;
    int rowsA, columnsA, rowsB, columnsB;
    bool printMatrix = false;
    Toperation *matrixA = nullptr;
    Toperation *matrixB = nullptr;

    auto fOptionChecker = std::find(optionsCmd.begin(), optionsCmd.end(), "-f");
    auto rOptionChecker = std::find(optionsCmd.begin(), optionsCmd.end(), "-r");
    auto gOptionChecker = std::find(optionsCmd.begin(), optionsCmd.end(), "-g");
    if (std::find(optionsCmd.begin(), optionsCmd.end(), "-h") != optionsCmd.end() || optionsCmd.size() == 0 || (fOptionChecker == optionsCmd.end() && rOptionChecker == optionsCmd.end()))
    {
        cout << "Uso:\tLas opciones -f y -r no se pueden usar a la vez" << endl;
        cout << "\t-h\tMuestra la ayuda" << endl;
        cout << "\t-p\t(Opcional) Muestra las matrices por pantalla" << endl;
        cout << "\t-f\tLas matrices son leidas de ficheros de texto: -f f1.txt f2.txt" << endl;
        cout << "\t-r\tLas matrices son generadas de forma aleatoria(m n k indican el tamaño de las matrices. bl bu indican de que numero a que numero se genera la matrix .Todos son numeros enteros) -r m n k " << endl;
        cout << "\t-g\t(Opcional) Indica el número de gpus que van a ser usadas. Pueden ser menos de las que hay en el sistema o más y se usaran algunas varias veces para simular ese número. Ejemplo -g 4" << endl;
        return;
    }
    if (std::find(optionsCmd.begin(), optionsCmd.end(), "-p") != optionsCmd.end())
    {
        printMatrix = true;
    }
    if (gOptionChecker != optionsCmd.end())
    {
        int gPosition = std::distance(optionsCmd.begin(), gOptionChecker);
        gpuSizeWorldArgument = atoi(optionsCmd[gPosition + 1].c_str());
    }
    else
    {
        gpuSizeWorldArgument = -1;
    }

    if (fOptionChecker != optionsCmd.end() && rOptionChecker != optionsCmd.end())
    {
        cout << "Los parametros -f y -r no se pueden usar a la vez" << endl;
        return;
    }
    if (fOptionChecker != optionsCmd.end())
    {
        int fPosition = std::distance(optionsCmd.begin(), fOptionChecker);
        matrixA = MatrixUtilities<Toperation>::ReadOrGenerateRandomMatrix(false, optionsCmd[fPosition + 1].c_str(), rowsA, columnsA, -1, -1);
        matrixB = MatrixUtilities<Toperation>::ReadOrGenerateRandomMatrix(false, optionsCmd[fPosition + 2].c_str(), rowsB, columnsB, -1, -1);
    }

    if (rOptionChecker != optionsCmd.end())
    {
        int rPosition = std::distance(optionsCmd.begin(), rOptionChecker);
        rowsA = atoi(optionsCmd[rPosition + 1].c_str());
        columnsA = atoi(optionsCmd[rPosition + 2].c_str());
        rowsB = atoi(optionsCmd[rPosition + 2].c_str());
        columnsB = atoi(optionsCmd[rPosition + 3].c_str());
        matrixA = MatrixUtilities<Toperation>::ReadOrGenerateRandomMatrix(true, "", rowsA, columnsA, atoi(optionsCmd[rPosition + 4].c_str()), atoi(optionsCmd[rPosition + 5].c_str()));
        matrixB = MatrixUtilities<Toperation>::ReadOrGenerateRandomMatrix(true, "", rowsB, columnsB, atoi(optionsCmd[rPosition + 4].c_str()), atoi(optionsCmd[rPosition + 5].c_str()));
    }

    NcclMultiplicationEnvironment<Toperation> ncclMultEnv = NcclMultiplicationEnvironment<Toperation>(gpuSizeWorldArgument, gpuRoot, opt, printMatrix);
    MatrixMain<Toperation> ma = MatrixMain<Toperation>(&ncclMultEnv, "A", rowsA, columnsA, matrixA);
    MatrixMain<Toperation> mb = MatrixMain<Toperation>(&ncclMultEnv, "B", rowsB, columnsB, matrixB);
    std::cout<<"Comienza el cálculo distribuido"<<std::endl;
    ctimer(&elapsedDistributed, &ucpuDistributed, &scpuDistributed);
    //Se puede usar de esta forma o de la otra.
    // MatrixMain<double> *mc=ncclMultEnv.performCalculations("A","B","C");
    ma =ma* mb;
    // ma*=ma;
    ctimer(&elapsedDistributed, &ucpuDistributed, &scpuDistributed);
    Toperation* distributedRes=ma.getHostMatrix();
    std::cout << "Tiempo del cálculo distribuido: " << elapsedDistributed << " segundos" << std::endl;
    if(printMatrix)
    {
        std::cout << "Resultado multigpu:" << std::endl;
        MatrixUtilities<Toperation>::printMatrix(ma.getRowsReal(), ma.getColumnsReal(), distributedRes);
    }
    

    //Una sola gpu
    CUDACHECK(cudaSetDevice(gpuRoot));
    cublasHandle_t handle;
    cudaStream_t streamWhole;
    CUDACHECK(cudaStreamCreate(&streamWhole));
    CUBLASCHECK(cublasCreate(&handle));
    Toperation *gpuWholeA, *gpuWholeB, *gpuWholeC,*hostResC;
    std::cout<<"Comienza el cálculo 1 gpu"<<std::endl;
    ctimer(&elapsedGpuNoDistributed, &ucpuGpuNoDistributed, &scpuGpuNoDistributed);
    gpuWholeA = MatrixUtilitiesCuda<Toperation>::cudaMatrixMemoryAllocation(rowsA, columnsA, &streamWhole);
    gpuWholeB = MatrixUtilitiesCuda<Toperation>::cudaMatrixMemoryAllocation(rowsB, columnsB, &streamWhole);
    gpuWholeC = MatrixUtilitiesCuda<Toperation>::cudaMatrixMemoryAllocation(rowsA, columnsB, &streamWhole);
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaMemcpy(gpuWholeA, matrixA, rowsA * columnsA * sizeof(Toperation), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(gpuWholeB, matrixB, rowsB * columnsB * sizeof(Toperation), cudaMemcpyHostToDevice));
    MatrixUtilitiesCuda<Toperation>::matrixCublasMultiplication(&handle, MultDouble, rowsA, columnsA, columnsB, gpuWholeA, gpuWholeB, gpuWholeC);
    CUDACHECK(cudaDeviceSynchronize());
    ctimer(&elapsedGpuNoDistributed, &ucpuGpuNoDistributed, &scpuGpuNoDistributed);
    hostResC=MatrixUtilities<Toperation>::matrixMemoryAllocation(rowsA,columnsB);
    CUDACHECK(cudaMemcpy(hostResC,gpuWholeC, rowsA * columnsB * sizeof(Toperation), cudaMemcpyDeviceToHost));
    std::cout << "Tiempo del cálculo 1 gpu: " << elapsedGpuNoDistributed << " segundos" << std::endl;
    if(printMatrix)
    {
        std::cout << "Resultado solo 1 gpu:" << std::endl;
        MatrixUtilitiesCuda<Toperation>::cudaPrintOneMatrixCall(rowsA, columnsB, gpuWholeC);
    }
    CUDACHECK(cudaStreamDestroy(streamWhole));
    CUBLASCHECK(cublasDestroy(handle));
    MatrixUtilitiesCuda<Toperation>::matrixFree(gpuWholeA);
    MatrixUtilitiesCuda<Toperation>::matrixFree(gpuWholeB);
    MatrixUtilitiesCuda<Toperation>::matrixFree(gpuWholeC);
    //Comparar si son iguales
    auto errores= MatrixUtilities<Toperation>::checkEqualityOfMatrices(hostResC,distributedRes,rowsA,columnsB);
    if(errores.size()==0)
    {
        std::cout<<"Las matrices son idénticas"<<std::endl;
    }else
    {
        std::cout<<"Las matrices no son iguales"<<std::endl;
    }
}

int main(int argc, char **argv)
{
    //////////////////////////////SE MARCARA CON //** las instrucciones necesarias para operar con la libreria y con //*** las opcionales
    
    OperationType opt = MultDouble;
    int i;
    cout << fixed;
    cout << setprecision(2);
    // MpiMultiplicationEnvironment<double> mpiMult = MpiMultiplicationEnvironment<double>(cpuRank, root, cpuSize, basicOperationType);//**

    //Lectura de los parametros de lanzamiento
    vector<string> optionsCmd;
    for (i = 0; i < argc; i++)
    {
        optionsCmd.push_back(string(argv[i]));
    }

    if (std::find(optionsCmd.begin(), optionsCmd.end(), "-tf") != optionsCmd.end())
    {
        opt = MultFloat;
        ejecucion<float>(optionsCmd, opt);
    }
    else
    {
        ejecucion<double>(optionsCmd, opt);
    }
}
