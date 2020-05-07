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
			printf("%.2f\t",matrix[IDX2C(i,j,rows)]);
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
             printf("%.2f\t",matrix[IDX2C(i,j,rows)]);
         }
         printf("\n");
     }
 }

//Métodos de la clase

template <class Toperation>
Toperation* MatrixUtilitiesCuda<Toperation>::cudaMatrixMemoryAllocationGPU(int rows, int columns,cudaStream_t *stream)
{
    Toperation* newMatrix;
    CUDACHECK(cudaMalloc ((void**)&newMatrix,rows*columns*sizeof(Toperation)));
    CUDACHECK(cudaMemsetAsync(newMatrix, 0, sizeof(Toperation)*rows*columns,*stream));
    return newMatrix;
}

template <class Toperation>
void MatrixUtilitiesCuda<Toperation>::matrixFreeGPU(Toperation *matrix)
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
void MatrixUtilitiesCuda<Toperation>::matrixCublasMultiplication(cublasHandle_t* handler,OperationType opt,int rowsA, int columnsAorRowsB, int columnsB, Toperation *A, Toperation *B, Toperation *C,Toperation alpha,Toperation beta)
{
    if(opt==MultDouble)
    {
        double alphaArg=alpha,betaArg=beta;
        CUBLASCHECK(cublasDgemm(*handler, CUBLAS_OP_N, CUBLAS_OP_N, rowsA, columnsB, columnsAorRowsB, &alphaArg, (double*)A, rowsA, (double*)B, columnsAorRowsB, &betaArg, (double*)C, rowsA));
    }else
    {
        float alphaArg=alpha,betaArg=beta;
        CUBLASCHECK(cublasSgemm(*handler, CUBLAS_OP_N, CUBLAS_OP_N, rowsA, columnsB, columnsAorRowsB, &alphaArg, (float*)A, rowsA, (float*)B, columnsAorRowsB, &betaArg, (float*)C, rowsA));
    }
}

template <class Toperation>
void MatrixUtilitiesCuda<Toperation>::axpyCublas(cublasHandle_t* handler,OperationType opt,int numberOfElementsToOperate, Toperation *X,Toperation *Y,Toperation alpha,Toperation strideX,Toperation strideY)
{
    if(opt==MultDouble)
    {
        const double alphaArg=alpha;
        CUBLASCHECK(cublasDaxpy(*handler, numberOfElementsToOperate,&alphaArg,(const double*)X, strideX,(double*)Y, strideY));
    }else
    {
        const float alphaArg=alpha;
        CUBLASCHECK(cublasSaxpy(*handler, numberOfElementsToOperate,&alphaArg,(const float*)X, strideX,(float* )Y, strideY));
    }
}

template <class Toperation>
void MatrixUtilitiesCuda<Toperation>::scalarCublas(cublasHandle_t* handler,OperationType opt,int rows, int columns, Toperation *X,Toperation alpha,Toperation strideX)
{
    if(opt==MultDouble)
    {
        const double alphaArg=alpha;
        CUBLASCHECK(cublasDscal(*handler, rows*columns,&alphaArg,(double*)X, strideX));
    }else
    {
        const float alphaArg=alpha;
        CUBLASCHECK(cublasSscal(*handler, rows*columns,&alphaArg,(float*)X, strideX));
    }
}

template <class Toperation>
void MatrixUtilitiesCuda<Toperation>::sumCublas(cublasHandle_t *handler, OperationType opt, int n,int strideX, Toperation *X, Toperation *result)
{
    if(opt==MultDouble)
    {
        CUBLASCHECK(cublasDasum(*handler, n,(double*)X,strideX,(double*) result));
    }else
    {
        CUBLASCHECK(cublasSasum(*handler, n,(float*)X,strideX, (float*)result));
    }
}

template <class Toperation>
Toperation* MatrixUtilitiesCuda<Toperation>::GenerateRandomMatrixGPU(int rows, int columns,OperationType opt)
{
    CUDACHECK(cudaSetDevice(0));
    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));
    unsigned long long numberOfElements=rows*columns;
    Toperation *matrixHost = MatrixUtilitiesCuda<Toperation>::matrixMemoryAllocationCPU(rows, columns);
    Toperation *matrixGpu=MatrixUtilitiesCuda<Toperation>::cudaMatrixMemoryAllocationGPU(rows,columns,&stream);
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
    matrixFreeGPU(matrixGpu);
    return matrixHost;
}

template <class Toperation>
void MatrixUtilitiesCuda<Toperation>::printMatrix(int rows, int columns, Toperation *M)
{
    int i, j, matrixIndex;
    for (i = 0; i < rows; ++i)
    {
        for (j = 0; j < columns; ++j)
        {
            matrixIndex = matrixCalculateIndex(rows,columns, i, j);
            std::cout << M[matrixIndex] << "\t";
        }
        std::cout << std::endl;
    }
}

template <class Toperation>
double MatrixUtilitiesCuda<Toperation>::checkEqualityOfMatrices(Toperation *A, Toperation *B, int rows, int columns)
{
    double normA=0,normB=0;
    int i, j;
    double elementA,elementB;
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < columns; j++)
        {
            elementA=A[IDX2C(i,j,rows)];
            normA+=(elementA*elementA);
            elementB=B[IDX2C(i,j,rows)];
            normB+=(elementB*elementB);
        }
    }
    normA=sqrt(normA);normB=sqrt(normB);
    double error = fabs(normA-normB)/normA;
    return sqrt(error);
}

template <class Toperation>
bool MatrixUtilitiesCuda<Toperation>::canMultiply(int columnsA, int rowsB)
{
    return columnsA == rowsB;
}

template <class Toperation>
OperationProperties MatrixUtilitiesCuda<Toperation>::getMeshAndMatrixSize(int rowsA, int columnsA, int rowsB, int columnsB, int cpuSize)
{
    OperationProperties res;

    //Se calculan todas las posibilidadades y se selecciona la que mas cpus use y menos 0 contenga de esas opciones, Solo se añaden elementos validos(ninguno con meshDimension 1 o 0)
    int i, j;
    std::vector<OperationProperties> allOp;
    std::vector<OperationProperties> sameCpuSizeOp;
    for (i = 2; i < cpuSize - 1; i++)
    {
        for (j = i; j * i <= cpuSize; j++)
        {
            OperationProperties opRow = calculateNonEqualMesh(rowsA, rowsB, columnsB, i, j, true);
            OperationProperties opColumn = calculateNonEqualMesh(rowsA, rowsB, columnsB, i, j, false);
            if (opRow.candidate)
            {
                allOp.push_back(opRow);
            }
            if (opColumn.candidate)
            {
                allOp.push_back(opColumn);
            }
        }
    }
    sort(begin(allOp), end(allOp), [](OperationProperties op1, OperationProperties op2) {
        if (op1.gpuSize != op2.gpuSize)
        {
            return op1.gpuSize > op2.gpuSize;
        }
        return op1.numberOf0 < op2.numberOf0;
    });
    res = allOp[0];

    return res;
}

template <class Toperation>
OperationProperties MatrixUtilitiesCuda<Toperation>::calculateNonEqualMesh(int rowsA, int columnsAorRowsB, int columnsB, int nCpusMesh1, int nCpusMesh2, bool isMeshRow)
{
    OperationProperties res;
    if (isMeshRow)
    {
        res.meshRowSize = nCpusMesh1;
        res.meshColumnSize = nCpusMesh2;
    }
    else
    {
        res.meshColumnSize = nCpusMesh1;
        res.meshRowSize = nCpusMesh2;
    }
    res.gpuSize = res.meshRowSize * res.meshColumnSize;
    res.rowsA = ceil(rowsA / (float)res.meshRowSize) * res.meshRowSize;
    res.columnsAorRowsB = ceil(columnsAorRowsB / (float)res.meshColumnSize) * res.meshColumnSize;
    res.columnsB = ceil(columnsB / (float)res.meshColumnSize) * res.meshColumnSize;
    int numberOf0atA = (res.rowsA * res.columnsAorRowsB) - (rowsA * columnsAorRowsB);
    int numberOf0atB = (res.columnsB * res.columnsAorRowsB) - (columnsAorRowsB * columnsB);
    res.numberOf0 = numberOf0atA + numberOf0atB;
    res.blockColumnSizeA = res.columnsAorRowsB / res.meshColumnSize;
    res.blockRowSizeB = res.blockColumnSizeA;
    res.blockRowSizeA = res.rowsA / res.meshRowSize;
    res.blockColumnSizeB = res.columnsB / res.meshColumnSize;
    res.candidate = res.meshColumnSize > 1 && res.meshRowSize > 1;
    return res;
}

template <class Toperation>
OperationProperties MatrixUtilitiesCuda<Toperation>::getMeshAndMatrixSizeFromOneDistributedMatrix(int rowsA, int columnsA, int rowsB, int columnsB, int meshRowSize,int meshColumnSize,bool isAAlreadyDistributed)
{
    OperationProperties res;
    res.meshRowSize=meshRowSize;
    res.meshColumnSize=meshColumnSize;
    if(isAAlreadyDistributed)
    {
        res.rowsA=rowsA;
        res.columnsAorRowsB=columnsA;
        res.columnsB = ceil(columnsB / (float)res.meshColumnSize) * res.meshColumnSize;
        res.blockRowSizeA = res.rowsA / res.meshRowSize;
        res.blockColumnSizeA = res.columnsAorRowsB / res.meshColumnSize;
        res.blockRowSizeB = res.blockColumnSizeA;
        res.blockColumnSizeB = res.columnsB / res.meshColumnSize;
    }else
    {
        res.columnsB=columnsB;
        res.columnsAorRowsB=rowsB;
        res.rowsA=ceil(rowsA / (float)res.meshRowSize) * res.meshRowSize;
        res.blockRowSizeA = res.rowsA / res.meshRowSize;
        res.blockColumnSizeA = res.columnsAorRowsB / res.meshColumnSize;
        res.blockRowSizeB = res.blockColumnSizeA;
        res.blockColumnSizeB = res.columnsB / res.meshColumnSize;
    }
    
    return res;
}

template <class Toperation>
Toperation *MatrixUtilitiesCuda<Toperation>::matrixMemoryAllocationCPU(int rows, int columns)
{
    Toperation *matrix = (Toperation *)calloc(rows * columns, sizeof(Toperation));
    return matrix;
}

template <class Toperation>
void MatrixUtilitiesCuda<Toperation>::matrixFreeCPU(Toperation *matrix)
{
    free(matrix);
}

template <class Toperation>
unsigned long long MatrixUtilitiesCuda<Toperation>::matrixCalculateIndex(int rowSize, int columnSize, int rowIndex, int columnIndex)
{
    return IDX2C(rowIndex,columnIndex,rowSize);
    // return columnSize * rowIndex + columnIndex;
}

template <class Toperation>
Toperation *MatrixUtilitiesCuda<Toperation>::ReadOrGenerateRandomMatrix(bool isRandom, const char *fileName, int &rows, int &columns, int boundLower, int boundUpper)
{
    unsigned long long i, j, matrixIndex;
    std::ifstream file;
    if (!isRandom)
    {
        file.open(fileName);
        file >> rows >> columns;
    }
    //Configuracion del generador de numeros por si se genera una matriz random
    std::random_device rd;
    std::mt19937 eng(rd());
    std::uniform_real_distribution<> distr(boundLower, boundUpper);
    Toperation *matrix = MatrixUtilitiesCuda<Toperation>::matrixMemoryAllocationCPU(rows, columns);
    //Bucle de generacion o lectura de la matrizs
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < columns; j++)
        {
            matrixIndex = MatrixUtilitiesCuda<Toperation>::matrixCalculateIndex(rows,columns, i, j);
            if (isRandom)
            {
                matrix[matrixIndex] = distr(eng);
            }
            else
            {
                file >> matrix[matrixIndex];
            }
        }
    }
    if (!isRandom)
    {
        file.close();
    }
    return matrix;
}

template <class Toperation>
void MatrixUtilitiesCuda<Toperation>::axpyBlas(OperationType opt, int numberOfElementsToOperate, Toperation *X, Toperation *Y, Toperation alpha, Toperation strideX, Toperation strideY)
{
    if(opt==MultDouble)
    {
        const double alphaArg=alpha;
        cblas_daxpy(numberOfElementsToOperate,alphaArg,(const double*)X, strideX,(double*)Y, strideY);
    }else
    {
        const float alphaArg=alpha;
        cblas_saxpy(numberOfElementsToOperate,alphaArg,(const float*)X, strideX,(float* )Y, strideY);
    }
}

template <class Toperation>
Toperation MatrixUtilitiesCuda<Toperation>::maximumBlas(OperationType opt, int numberOfElementsToOperate, Toperation *X, Toperation strideX)
{
    CBLAS_INDEX indexMax;
    if(opt==MultDouble)
    {
        indexMax=cblas_idamax(numberOfElementsToOperate, (const double*)X,strideX);
    }else
    {
        indexMax=cblas_isamax(numberOfElementsToOperate, (const float*)X,strideX);
    }
    return X[indexMax];
}

template class MatrixUtilitiesCuda<double>;
template class MatrixUtilitiesCuda<float>;
