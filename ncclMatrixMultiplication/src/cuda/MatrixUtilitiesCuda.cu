#include "MatrixUtilitiesCuda.cuh"
#include "ErrorCheckingCuda.cuh"


__global__ void
cudaPrintMatrix(int rows,int columns,double* matrix)
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

template <class Toperation>
int MatrixUtilitiesCuda<Toperation>::matrixCalculateIndex(int rowSize, int columnSize, int rowIndex, int columnIndex)
{
    return IDX2C(rowIndex,columnIndex,rowSize);
    // return columnSize * rowIndex + columnIndex;
}
template <class Toperation>
Toperation* MatrixUtilitiesCuda<Toperation>::cudaMatrixMemoryAllocation(int rows, int columns,cudaStream_t *stream)
{
    Toperation* newMatrix;
    CUDACHECK(cudaMalloc ((void**)&newMatrix,rows*columns*sizeof(double)));
    CUDACHECK(cudaMemsetAsync(newMatrix, 0, sizeof(double)*rows*columns,*stream));
    return newMatrix;
}

template <class Toperation>
int MatrixUtilitiesCuda<Toperation>::getRealGpuId(int gpuRankOperation,int gpuSizeSystem)
{
    return gpuRankOperation%gpuSizeSystem;
}

template <class Toperation>
void MatrixUtilitiesCuda<Toperation>::cudaPrintOneMatrixCall(int rows,int columns,Toperation* matrix)
{
    cudaPrintMatrix<<<1,1,1>>>(rows,columns,(double*)matrix);
    CUDACHECK(cudaDeviceSynchronize());
}

template <class Toperation>
void MatrixUtilitiesCuda<Toperation>::cudaDebugMatrixDifferentGpus(int gpuRank, int rows, int columns, Toperation *M, std::string extraMessage)
{
    usleep(gpuRank * 1000);
    std::cout << "Parte del gpuWorker: " << gpuRank << " " << extraMessage << std::endl;
    MatrixUtilitiesCuda::cudaPrintOneMatrixCall(rows, columns, M);
}

template <class Toperation>
void MatrixUtilitiesCuda<Toperation>::cudaDebugMatricesLocalDifferentGpuWorkers(int gpuSize, int rows, int columns, std::vector<GpuWorker<Toperation>*> gpuWorkers)
{
    unsigned int gpuRank,j;
    for (gpuRank = 0; gpuRank < gpuWorkers.size(); gpuRank++)
    {
        for(j=0;j<gpuWorkers[gpuRank]->getMatricesLocal().size();j++)
        {
            //W.I.P CREO QUE EL CALCULO DEL TOSTRING ESTA MAL
            std::string msg = " Matriz local: " + std::to_string((gpuRank + (j * gpuSize)));
            MatrixUtilitiesCuda::cudaDebugMatrixDifferentGpus(gpuRank, rows, columns, gpuWorkers[gpuRank]->getMatricesLocal()[j], msg);
        }
    }
}


template class MatrixUtilitiesCuda<double>;