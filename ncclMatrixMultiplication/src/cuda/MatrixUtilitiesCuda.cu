#include "MatrixUtilitiesCuda.cuh"
#include "ErrorCheckingCuda.cuh"


__global__ void
cudaPrintMatrix(int rows,int columns,double* matrix)
{
	for(int i =0;i<rows;i++)
	{
		for(int j=0;j<columns;j++)
		{
			printf("%.2lf\t",matrix[IDX2C(i,j,rows)]);
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
Toperation* MatrixUtilitiesCuda<Toperation>::cudaMatrixMemoryAllocation(int rows, int columns)
{
    Toperation* newMatrix;
    CUDACHECK(cudaMalloc ((void**)&newMatrix,rows*columns*sizeof(double)));
    CUDACHECK(cudaMemsetAsync(newMatrix, 0, sizeof(double)*rows*columns,0));
    return newMatrix;
}

template <class Toperation>
int MatrixUtilitiesCuda<Toperation>::getRealGpuId(int gpuRankOperation,int gpuSizeSystem)
{
    return gpuRankOperation%gpuSizeSystem;
}

template <class Toperation>
void MatrixUtilitiesCuda<Toperation>::cudaPrintMatrixCall(int rows,int columns,double* matrix)
{
    cudaPrintMatrix<<<1,1,1>>>(rows,columns,matrix);
}

template class MatrixUtilitiesCuda<double>;