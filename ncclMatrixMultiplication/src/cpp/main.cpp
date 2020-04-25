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

/**
 * @brief Método que realiza una ejecución de la librería
 * 
 * SE MARCARA CON //** las instrucciones necesarias para operar con la libreria y con //*** las opcionales
 * @tparam Toperation , tipo de la matriz(double,float)  (Tiene que concordar con opt)
 * @param optionsCmd , opciones leidas de la entrada
 * @param opt , tipo de operación. MultDouble|MultFloat (Tiene que concordar con Toperation)
 */
template <class Toperation>
void ejecucion(vector<string> optionsCmd, OperationType opt)
{
    //////////////////////////////
    int i,rowsA, columnsA, rowsB, columnsB,rowsC,columnsC,gpuSizeWorldArgument, cpuOperationsSize,gpuRoot=0,iterations=10;
    double elapsedDistributed, ucpuDistributed, scpuDistributed,elapsedGpuNoDistributed, ucpuGpuNoDistributed, scpuGpuNoDistributed;
    bool printMatrix = false;
    Toperation *matrixA = nullptr;
    Toperation *matrixB = nullptr;
    Toperation *matrixAAux1Gpu = nullptr;
    Toperation *matrixBAux1Gpu = nullptr;
    Toperation *distributedRes=nullptr;
    Toperation *hostResC=nullptr;

    auto fOptionChecker = std::find(optionsCmd.begin(), optionsCmd.end(), "-f");
    auto rOptionChecker = std::find(optionsCmd.begin(), optionsCmd.end(), "-r");
    auto rgOptionChecker = std::find(optionsCmd.begin(), optionsCmd.end(), "-rg");
    auto gOptionChecker = std::find(optionsCmd.begin(), optionsCmd.end(), "-g");
    auto itOptionChecker = std::find(optionsCmd.begin(), optionsCmd.end(), "-it");

    if (std::find(optionsCmd.begin(), optionsCmd.end(), "-h") != optionsCmd.end() || optionsCmd.size() == 0 || (fOptionChecker == optionsCmd.end() && rOptionChecker == optionsCmd.end() && rgOptionChecker == optionsCmd.end()))
    {
        cout << "Uso:\tLas opciones -f y -r no se pueden usar a la vez" << endl;
        cout << "\t-h\tMuestra la ayuda" << endl;
        cout << "\t-p\t(Opcional) Muestra las matrices por pantalla" << endl;
        cout << "\t-it num\t(Opcional) Num es el número de iteraciones a realizar" << endl;
        cout << "\t-f\tLas matrices son leidas de ficheros de texto: -f f1.txt f2.txt" << endl;
        cout << "\t-r|-rg\tLas matrices son generadas de forma aleatoria(m n k indican el tamaño de las matrices). Si es generación mediante cpu -r, mediante gpu -rg. -r m n k " << endl;
        cout << "\t-g\t(Opcional) Indica el número de gpus que van a ser usadas. Pueden ser menos de las que hay en el sistema o más y se usaran algunas varias veces para simular ese número. Automaticamente se usaran 4 gpus lógicas como mínimo. Ejemplo -g 4" << endl;
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
    if (itOptionChecker != optionsCmd.end())
    {
        int itPosition = std::distance(optionsCmd.begin(), itOptionChecker);
        iterations = atoi(optionsCmd[itPosition + 1].c_str());
    }
    else
    {
        gpuSizeWorldArgument = -1;
    }

    if ((fOptionChecker != optionsCmd.end() && rOptionChecker != optionsCmd.end() &&rgOptionChecker != optionsCmd.end() )|| (rgOptionChecker != optionsCmd.end() && rOptionChecker != optionsCmd.end()))
    {
        cout << "Los parametros -f, -rg y -r no se pueden usar a la vez" << endl;
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
        matrixA = MatrixUtilities<Toperation>::ReadOrGenerateRandomMatrix(true, "", rowsA, columnsA, 0, 1);
        matrixB = MatrixUtilities<Toperation>::ReadOrGenerateRandomMatrix(true, "", rowsB, columnsB, 0, 1);
    }
    if (rgOptionChecker != optionsCmd.end())
    {
        int rgPosition = std::distance(optionsCmd.begin(), rgOptionChecker);
        rowsA = atoi(optionsCmd[rgPosition + 1].c_str());
        columnsA = atoi(optionsCmd[rgPosition + 2].c_str());
        rowsB = atoi(optionsCmd[rgPosition + 2].c_str());
        columnsB = atoi(optionsCmd[rgPosition + 3].c_str());
        matrixA = MatrixUtilitiesCuda<Toperation>::GenerateRandomMatrix(rowsA, columnsA,opt);
        matrixB = MatrixUtilitiesCuda<Toperation>::GenerateRandomMatrix(rowsB, columnsB,opt);
    }
    //Copiar las matrices del host nuevamente para el cálculo de 1 gpu
    matrixAAux1Gpu=MatrixUtilities<Toperation>::matrixMemoryAllocation(rowsA, columnsA);
    matrixBAux1Gpu=MatrixUtilities<Toperation>::matrixMemoryAllocation(rowsB, columnsB);
    memcpy(matrixAAux1Gpu,matrixA,sizeof(Toperation)*rowsA*columnsA);
    memcpy(matrixBAux1Gpu,matrixB,sizeof(Toperation)*rowsB*columnsB);

    //Cálculo Multigpu
    {
        NcclMultiplicationEnvironment<Toperation> ncclMultEnv = NcclMultiplicationEnvironment<Toperation>(gpuSizeWorldArgument, gpuRoot, opt, printMatrix);
        MatrixMain<Toperation> ma = MatrixMain<Toperation>(&ncclMultEnv, "A", rowsA, columnsA, matrixA);
        MatrixMain<Toperation> mp = MatrixMain<Toperation>(&ncclMultEnv, "P", rowsA, columnsA, matrixA);
        MatrixMain<Toperation> mb = MatrixMain<Toperation>(&ncclMultEnv, "B", rowsB, columnsB, matrixB);

        std::cout<<"Comienza el cálculo distribuido. Iteraciones: "<<iterations<<std::endl;
        ctimer(&elapsedDistributed, &ucpuDistributed, &scpuDistributed);
        for(i=0;i<iterations;i++)
        {
            //Se puede usar de esta forma o de la otra.
            // ma =ma* ma;
            // ma*=mp; 
        }
        MatrixMain<Toperation> mc=ma*mb;
        // mc=3*ma;
        ma+=1;
        // mc=ma;
        ctimer(&elapsedDistributed, &ucpuDistributed, &scpuDistributed);
        distributedRes=ma.getHostMatrix();
        rowsC=mc.getRowsReal();
        columnsC=mc.getColumnsReal();
    }
    std::cout << "Tiempo del cálculo distribuido: " << elapsedDistributed << " segundos" << std::endl;
    if(printMatrix)
    {
        std::cout << "Resultado multigpu:" << std::endl;
        MatrixUtilities<Toperation>::printMatrix(rowsC, columnsC, distributedRes);
    }
    

    //Una sola gpu
    {
        CUDACHECK(cudaSetDevice(gpuRoot));
        cublasHandle_t handle;
        cudaStream_t streamWhole;
        CUDACHECK(cudaStreamCreate(&streamWhole));
        CUBLASCHECK(cublasCreate(&handle));
        Toperation *gpuWholeA, *gpuWholeB, *gpuWholeAux,*gpuWholeP,*gpuWholeRes;
        std::cout<<"Comienza el cálculo 1 gpu"<<std::endl;
        ctimer(&elapsedGpuNoDistributed, &ucpuGpuNoDistributed, &scpuGpuNoDistributed);
        gpuWholeA = MatrixUtilitiesCuda<Toperation>::cudaMatrixMemoryAllocation(rowsA, columnsA, &streamWhole);
        gpuWholeP = MatrixUtilitiesCuda<Toperation>::cudaMatrixMemoryAllocation(rowsA, columnsA, &streamWhole);
        gpuWholeB = MatrixUtilitiesCuda<Toperation>::cudaMatrixMemoryAllocation(rowsB, columnsB, &streamWhole);
        gpuWholeAux = MatrixUtilitiesCuda<Toperation>::cudaMatrixMemoryAllocation(rowsA, columnsA, &streamWhole);
        gpuWholeRes = MatrixUtilitiesCuda<Toperation>::cudaMatrixMemoryAllocation(rowsA, columnsB, &streamWhole);
        CUDACHECK(cudaDeviceSynchronize());
        CUDACHECK(cudaMemcpy(gpuWholeA, matrixAAux1Gpu, rowsA * columnsA * sizeof(Toperation), cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemcpy(gpuWholeP, matrixAAux1Gpu, rowsA * columnsA * sizeof(Toperation), cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemcpy(gpuWholeB, matrixBAux1Gpu, rowsB * columnsB * sizeof(Toperation), cudaMemcpyHostToDevice));
        for(i=0;i<iterations;i++)
        {
            MatrixUtilitiesCuda<Toperation>::matrixCublasMultiplication(&handle, opt, rowsA, columnsA, columnsA, gpuWholeA, gpuWholeP, gpuWholeAux,1.0,0.0);
            CUDACHECK(cudaDeviceSynchronize());
            CUDACHECK(cudaMemcpy(gpuWholeA, gpuWholeAux, rowsA * columnsA * sizeof(Toperation), cudaMemcpyDeviceToDevice));
        }
        MatrixUtilitiesCuda<Toperation>::matrixCublasMultiplication(&handle, opt, rowsA, columnsA, columnsA, gpuWholeA, gpuWholeB, gpuWholeRes,1.0,0.0);
        CUDACHECK(cudaDeviceSynchronize());
        ctimer(&elapsedGpuNoDistributed, &ucpuGpuNoDistributed, &scpuGpuNoDistributed);
        hostResC=MatrixUtilities<Toperation>::matrixMemoryAllocation(rowsA,columnsB);
        CUDACHECK(cudaMemcpy(hostResC,gpuWholeRes, rowsA * columnsB * sizeof(Toperation), cudaMemcpyDeviceToHost));
        CUDACHECK(cudaStreamDestroy(streamWhole));
        CUBLASCHECK(cublasDestroy(handle));
        MatrixUtilitiesCuda<Toperation>::matrixFree(gpuWholeA);
        MatrixUtilitiesCuda<Toperation>::matrixFree(gpuWholeB);
        MatrixUtilitiesCuda<Toperation>::matrixFree(gpuWholeAux);
        MatrixUtilitiesCuda<Toperation>::matrixFree(gpuWholeP);
        MatrixUtilitiesCuda<Toperation>::matrixFree(gpuWholeRes);

    }
    std::cout << "Tiempo del cálculo 1 gpu: " << elapsedGpuNoDistributed << " segundos" << std::endl;
    if(printMatrix)
    {
        std::cout << "Resultado solo 1 gpu:" << std::endl;
        // MatrixUtilities<Toperation>::printMatrix(rowsC, columnsC, hostResC);
    }
    
    //Comparar si son iguales
    auto errores= MatrixUtilities<Toperation>::checkEqualityOfMatrices(hostResC,distributedRes,rowsA,columnsB);
    if(errores.size()==0)
    {
        std::cout<<"Las matrices son idénticas"<<std::endl;
    }else
    {
        std::cout<<"Las matrices no son iguales"<<std::endl;
    }
    MatrixUtilities<Toperation>::matrixFree(matrixAAux1Gpu);
    MatrixUtilities<Toperation>::matrixFree(matrixBAux1Gpu);
    MatrixUtilities<Toperation>::matrixFree(hostResC);
}

int main(int argc, char **argv)
{
    int i;
    cout << fixed;
    cout << setprecision(2);
    //Lectura de los parametros de lanzamiento
    vector<string> optionsCmd;
    for (i = 0; i < argc; i++)
    {
        optionsCmd.push_back(string(argv[i]));
    }

    if (std::find(optionsCmd.begin(), optionsCmd.end(), "-tf") != optionsCmd.end())
    {
        ejecucion<float>(optionsCmd, MultFloat);
    }
    else
    {
        ejecucion<double>(optionsCmd, MultDouble);
    }
}
