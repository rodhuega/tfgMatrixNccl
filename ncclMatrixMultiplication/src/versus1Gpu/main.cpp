#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <algorithm>

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
    int i,rowsA, columnsA,rowsC,columnsC,gpuSizeWorldArgument,gpuRoot=0,iterations=10;
    double elapsedDistributed, ucpuDistributed, scpuDistributed,elapsedGpuNoDistributed, ucpuGpuNoDistributed, scpuGpuNoDistributed;
    bool printMatrix = false;
    Toperation *matrixA = nullptr;
    Toperation *matrixAAux1Gpu = nullptr;
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
    }else
    {
        gpuSizeWorldArgument = -1;
    }

    if (itOptionChecker != optionsCmd.end())
    {
        int itPosition = std::distance(optionsCmd.begin(), itOptionChecker);
        iterations = atoi(optionsCmd[itPosition + 1].c_str());
    }
    

    if ((fOptionChecker != optionsCmd.end() && rOptionChecker != optionsCmd.end() &&rgOptionChecker != optionsCmd.end() )|| (rgOptionChecker != optionsCmd.end() && rOptionChecker != optionsCmd.end()))
    {
        cout << "Los parametros -f, -rg y -r no se pueden usar a la vez" << endl;
        return;
    }
    if (fOptionChecker != optionsCmd.end())
    {
        int fPosition = std::distance(optionsCmd.begin(), fOptionChecker);
        matrixA = MatrixUtilitiesCuda<Toperation>::ReadOrGenerateRandomMatrix(false, optionsCmd[fPosition + 1].c_str(), rowsA, columnsA, -1, -1);
    }

    if (rOptionChecker != optionsCmd.end())
    {
        int rPosition = std::distance(optionsCmd.begin(), rOptionChecker);
        rowsA = atoi(optionsCmd[rPosition + 1].c_str());
        columnsA = rowsA;
        matrixA = MatrixUtilitiesCuda<Toperation>::ReadOrGenerateRandomMatrix(true, "", rowsA, columnsA, 0, 1);
    }
    if (rgOptionChecker != optionsCmd.end())
    {
        int rgPosition = std::distance(optionsCmd.begin(), rgOptionChecker);
        rowsA = atoi(optionsCmd[rgPosition + 1].c_str());
        columnsA = rowsA;
        matrixA = MatrixUtilitiesCuda<Toperation>::GenerateRandomMatrixGPU(rowsA, columnsA,opt);
    }
    //Copiar las matrices del host nuevamente para el cálculo de 1 gpu
    matrixAAux1Gpu=MatrixUtilitiesCuda<Toperation>::matrixMemoryAllocationCPU(rowsA, columnsA);
    memcpy(matrixAAux1Gpu,matrixA,sizeof(Toperation)*rowsA*columnsA);

    //Cálculo Multigpu
    {
        std::vector<MatrixMain<Toperation>*> mats;
        NcclMultiplicationEnvironment<Toperation> ncclMultEnv = NcclMultiplicationEnvironment<Toperation>(gpuSizeWorldArgument, gpuRoot, opt, printMatrix);
        {
                MatrixMain<Toperation> *ma = new MatrixMain<Toperation>(&ncclMultEnv, rowsA, columnsA, matrixA);
                mats.push_back(ma);
        }
        // MatrixMain<Toperation> mb = MatrixMain<Toperation>(&ncclMultEnv, rowsA, columnsA);
        // MatrixMain<Toperation> ma = MatrixMain<Toperation>(&ncclMultEnv, rowsA,columnsA);
        // mb.setMatrixHostToFullValue(1);
        // ma.setMatrixHost(matrixA);
        std::cout<<"Comienza el cálculo distribuido. Iteraciones: "<<iterations<<std::endl;
        ctimer(&elapsedDistributed, &ucpuDistributed, &scpuDistributed);
        double norma1A=mats[0]->norm1();
                        MatrixMain<Toperation> *auxMa;
        for(i=0;i<iterations;i++)
        {
            //Se puede usar de esta forma o de la otra.
            // ma =ma* mp;
            {
                auxMa=&((*mats[0])*(*mats[mats.size()-1]));
                mats.push_back(auxMa);
            }
        }
        double e=2;
        int s=4;
        for( i=0;i<mats.size();i++) {
            std::cout<<i<<". Esta distribuida: "<<mats[i]->getIsDistributed()<<", esta en el host: "<<(*mats[i]).getIsMatrixHostHere() <<std::endl;;
            (*mats[i])/=pow( e, s*(i++));
        }
        // ma.axpy(2,ma);
        // ma=ma/10;
        // ma=3-ma;
        
        ctimer(&elapsedDistributed, &ucpuDistributed, &scpuDistributed);
        rowsC=mats[0]->getRowsReal();
        columnsC=mats[0]->getColumnsReal();
        distributedRes=MatrixUtilitiesCuda<Toperation>::matrixMemoryAllocationCPU(rowsC, columnsC);
        mats[4]->getHostMatrixInThisPointer(distributedRes);
    }
    // {
    //     std::vector<MatrixMain<Toperation>> mats;
    //     NcclMultiplicationEnvironment<Toperation> ncclMultEnv = NcclMultiplicationEnvironment<Toperation>(gpuSizeWorldArgument, gpuRoot, opt, printMatrix);
    //     {
    //             MatrixMain<Toperation> ma = MatrixMain<Toperation>(&ncclMultEnv, rowsA, columnsA, matrixA);
    //             mats.push_back(ma);
    //     }
    //     // MatrixMain<Toperation> mb = MatrixMain<Toperation>(&ncclMultEnv, rowsA, columnsA);
    //     // MatrixMain<Toperation> ma = MatrixMain<Toperation>(&ncclMultEnv, rowsA,columnsA);
    //     // mb.setMatrixHostToFullValue(1);
    //     // ma.setMatrixHost(matrixA);
    //     std::cout<<"Comienza el cálculo distribuido. Iteraciones: "<<iterations<<std::endl;
    //     ctimer(&elapsedDistributed, &ucpuDistributed, &scpuDistributed);
    //     double norma1A=mats[0].norm1();
    //     for(i=0;i<iterations;i++)
    //     {
    //         //Se puede usar de esta forma o de la otra.
    //         // ma =ma* mp;
    //         {
    //             mats.push_back(mats[0]*mats[mats.size()-1]);
    //         }
    //     }
    //     double e=2;
    //     int s=4;
    //     for( i=0;i<mats.size();i++) {
    //         std::cout<<i<<". Esta distribuida: "<<mats[i].getIsDistributed()<<", esta en el host: "<<mats[i].getIsMatrixHostHere() <<std::endl;;
    //         mats[i]/=pow( e, s*(i++));
    //     }
    //     // ma.axpy(2,ma);
    //     // ma=ma/10;
    //     // ma=3-ma;
        
    //     ctimer(&elapsedDistributed, &ucpuDistributed, &scpuDistributed);
    //     rowsC=mats[0].getRowsReal();
    //     columnsC=mats[0].getColumnsReal();
    //     distributedRes=MatrixUtilitiesCuda<Toperation>::matrixMemoryAllocationCPU(rowsC, columnsC);
    //     mats[4].getHostMatrixInThisPointer(distributedRes);
    // }
    std::cout << "Tiempo del cálculo distribuido: " << elapsedDistributed << " segundos" << std::endl;
    if(printMatrix)
    {
        std::cout << "Resultado multigpu:" << std::endl;
        MatrixUtilitiesCuda<Toperation>::printMatrix(rowsC, columnsC, distributedRes);
    }
    

    //Una sola gpu
    {
        CUDACHECK(cudaSetDevice(gpuRoot));
        cublasHandle_t handle;
        cudaStream_t streamWhole;
        CUDACHECK(cudaStreamCreate(&streamWhole));
        CUBLASCHECK(cublasCreate(&handle));
        Toperation *gpuWholeA,*gpuWholeRes;
        std::cout<<"Comienza el cálculo 1 gpu"<<std::endl;
        ctimer(&elapsedGpuNoDistributed, &ucpuGpuNoDistributed, &scpuGpuNoDistributed);
        gpuWholeA = MatrixUtilitiesCuda<Toperation>::cudaMatrixMemoryAllocationGPU(rowsA, columnsA, &streamWhole);
        gpuWholeRes = MatrixUtilitiesCuda<Toperation>::cudaMatrixMemoryAllocationGPU(rowsA, columnsA, &streamWhole);
        CUDACHECK(cudaDeviceSynchronize());
        CUDACHECK(cudaMemcpy(gpuWholeA, matrixAAux1Gpu, rowsA * columnsA * sizeof(Toperation), cudaMemcpyHostToDevice));
        for(i=0;i<iterations;i++)
        {
            MatrixUtilitiesCuda<Toperation>::matrixCublasMultiplication(&handle, opt, rowsA, columnsA, columnsA, gpuWholeA, gpuWholeA, gpuWholeRes,1.0,0.0);
            CUDACHECK(cudaDeviceSynchronize());
            CUDACHECK(cudaMemcpy(gpuWholeA, gpuWholeRes, rowsA * columnsA * sizeof(Toperation), cudaMemcpyDeviceToDevice));
        }
        CUDACHECK(cudaDeviceSynchronize());
        ctimer(&elapsedGpuNoDistributed, &ucpuGpuNoDistributed, &scpuGpuNoDistributed);
        hostResC=MatrixUtilitiesCuda<Toperation>::matrixMemoryAllocationCPU(rowsA,columnsA);
        CUDACHECK(cudaMemcpy(hostResC,gpuWholeRes, rowsA * columnsA * sizeof(Toperation), cudaMemcpyDeviceToHost));
        CUDACHECK(cudaStreamDestroy(streamWhole));
        CUBLASCHECK(cublasDestroy(handle));
        MatrixUtilitiesCuda<Toperation>::matrixFreeGPU(gpuWholeA);
        MatrixUtilitiesCuda<Toperation>::matrixFreeGPU(gpuWholeRes);
    }
    std::cout << "Tiempo del cálculo 1 gpu: " << elapsedGpuNoDistributed << " segundos" << std::endl;
    if(printMatrix)
    {
        std::cout << "Resultado solo 1 gpu:" << std::endl;
        // MatrixUtilitiesCuda<Toperation>::printMatrix(rowsC, columnsC, hostResC);
    }
    
    //Comparar si son iguales
    double error= MatrixUtilitiesCuda<Toperation>::checkEqualityOfMatrices(hostResC,distributedRes,rowsA,columnsA);
    std::cout<<"El error relativo es: "<<error<<std::endl;
    MatrixUtilitiesCuda<Toperation>::matrixFreeCPU(matrixAAux1Gpu);
    MatrixUtilitiesCuda<Toperation>::matrixFreeCPU(hostResC);
    std::cout<<"El speedup es de: "<<elapsedGpuNoDistributed/elapsedDistributed<<std::endl;

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
