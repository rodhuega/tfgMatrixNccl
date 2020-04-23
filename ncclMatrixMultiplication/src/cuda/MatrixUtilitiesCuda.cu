#include "MatrixUtilitiesCuda.cuh"

//Métodos Kernel de la gpu.

/**
 * @brief Kernel que imprime por pantalla la matriz alojada en la gpu en formato double.
 * 
 * @param rows , filas de la matriz
 * @param columns , columnas de la matriz
 * @param matrix , matriz de la gpu a mostrar
 */
__global__ void
cudaPrintMatrixDouble(int rows,int columns,double* matrix)
{
	for(int i =0;i<rows;i++)
	{
		for(int j=0;j<columns;j++)
		{
			printf("%.2f\t",matrix[IDX2CGPU(i,j,rows)]);
		}
		printf("\n");
	}
}
/**
 * @brief Kernel que imprime por pantalla la matriz alojada en la gpu en formato float.
 * 
 * @param rows , filas de la matriz
 * @param columns , columnas de la matriz
 * @param matrix , matriz de la gpu a mostrar
 */
 __global__ void
 cudaPrintMatrixFloat(int rows,int columns,float* matrix)
 {
     for(int i =0;i<rows;i++)
     {
         for(int j=0;j<columns;j++)
         {
             printf("%.2f\t",matrix[IDX2CGPU(i,j,rows)]);
         }
         printf("\n");
     }
 }

//Métodos de la clase

template <class Toperation>
Toperation* MatrixUtilitiesCuda<Toperation>::cudaMatrixMemoryAllocation(int rows, int columns,cudaStream_t *stream)
{
    Toperation* newMatrix;
    CUDACHECK(cudaMalloc ((void**)&newMatrix,rows*columns*sizeof(Toperation)));
    CUDACHECK(cudaMemsetAsync(newMatrix, 0, sizeof(Toperation)*rows*columns,*stream));
    return newMatrix;
}

template <class Toperation>
void MatrixUtilitiesCuda<Toperation>::matrixFree(Toperation *matrix)
{
    CUDACHECK(cudaFree(matrix));
}

template <class Toperation>
int MatrixUtilitiesCuda<Toperation>::getRealGpuId(int gpuRankOperation,int gpuSizeSystem)
{
    return gpuRankOperation%gpuSizeSystem;
}

template <class Toperation>
void MatrixUtilitiesCuda<Toperation>::cudaPrintOneMatrixCall(int rows,int columns,Toperation* matrix,OperationType opt)
{
    if(opt==MultDouble)
    {
        cudaPrintMatrixDouble<<<1,1,1>>>(rows,columns,(double*)matrix);
    }else
    {
        cudaPrintMatrixFloat<<<1,1,1>>>(rows,columns,(float*)matrix);
    }
    CUDACHECK(cudaDeviceSynchronize());
}

template <class Toperation>
void MatrixUtilitiesCuda<Toperation>::cudaDebugMatrixDifferentGpus(int gpuRank, int rows, int columns, Toperation *M, std::string extraMessage,OperationType opt)
{
    usleep(gpuRank * 1000);
    std::cout << "Parte del gpuWorker: " << gpuRank << " " << extraMessage << std::endl;
    MatrixUtilitiesCuda::cudaPrintOneMatrixCall(rows, columns, M,opt);
}

template <class Toperation>
void MatrixUtilitiesCuda<Toperation>::cudaDebugMatricesLocalDifferentGpuWorkers(int gpuSizeOperationWorld,int gpuSizeSystem, int rows, int columns, std::vector<GpuWorker<Toperation>*> gpuWorkers,OperationType opt)
{
    unsigned int gpuRank,j;
    for (gpuRank = 0; gpuRank < gpuWorkers.size(); gpuRank++)
    {
        int gpuRealId=MatrixUtilitiesCuda<Toperation>::getRealGpuId(gpuRank,gpuSizeSystem);
        CUDACHECK(cudaSetDevice(gpuRealId));
        for(j=0;j<gpuWorkers[gpuRank]->getMatricesLocal().size();j++)
        {
            //W.I.P CREO QUE EL CALCULO DEL TOSTRING ESTA MAL
            std::string msg =" Id gpu real: "+std::to_string(gpuWorkers[gpuRank]->getGpuRankSystem()) +" Matriz local: " + std::to_string((gpuRank + (j * gpuSizeOperationWorld)));
            MatrixUtilitiesCuda::cudaDebugMatrixDifferentGpus(gpuRank, rows, columns, gpuWorkers[gpuRank]->getMatricesLocal()[j], msg,opt);
        }
    }
}

template <class Toperation>
void MatrixUtilitiesCuda<Toperation>::matrixCublasMultiplication(cublasHandle_t* handler,OperationType opt,int rowsA, int columnsAorRowsB, int columnsB, Toperation *A, Toperation *B, Toperation *C,Toperation alfa,Toperation beta)
{
    if(opt==MultDouble)
    {
        double alfaArg=alfa,betaArg=beta;
        CUBLASCHECK(cublasDgemm(*handler, CUBLAS_OP_N, CUBLAS_OP_N, rowsA, columnsB, columnsAorRowsB, &alfaArg, (double*)A, rowsA, (double*)B, columnsAorRowsB, &betaArg, (double*)C, rowsA));
    }else
    {
        float alfaArg=alfa,betaArg=beta;
        CUBLASCHECK(cublasSgemm(*handler, CUBLAS_OP_N, CUBLAS_OP_N, rowsA, columnsB, columnsAorRowsB, &alfaArg, (float*)A, rowsA, (float*)B, columnsAorRowsB, &betaArg, (float*)C, rowsA));
    }
}

template <class Toperation>
Toperation* MatrixUtilitiesCuda<Toperation>::GenerateRandomMatrix(int rows, int columns,OperationType opt)
{
    CUDACHECK(cudaSetDevice(0));
    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));
    unsigned long long numberOfElements=rows*columns;
    Toperation *matrixHost = MatrixUtilities<Toperation>::matrixMemoryAllocation(rows, columns);
    Toperation *matrixGpu=cudaMatrixMemoryAllocation(rows,columns,&stream);
    curandGenerator_t generator;
    srand(time(NULL));
    int seed = rand();
    CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, seed));
    CUDACHECK(cudaStreamSynchronize(stream));
    if(opt==MultDouble)
    {
        CURAND_CALL(curandGenerateUniformDouble(generator, (double*)matrixGpu, numberOfElements));
    }else
    {
        CURAND_CALL(curandGenerateUniform(generator, (float*)matrixGpu, numberOfElements));
    }
    CUDACHECK(cudaMemcpyAsync(matrixHost,matrixGpu,numberOfElements*sizeof(Toperation),cudaMemcpyDeviceToHost,stream));
    CUDACHECK(cudaStreamSynchronize(stream));
    CURAND_CALL(curandDestroyGenerator(generator));
    CUDACHECK(cudaStreamDestroy(stream));
    matrixFree(matrixGpu);
    return matrixHost;
}


template class MatrixUtilitiesCuda<double>;
template class MatrixUtilitiesCuda<float>;
