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
    // CUDACHECK(cudaEventRecord(startMalloc1[i]));
    // CUDACHECK(cudaMalloc ((void**)&gpusInfo[i]->matrixDeviceA,rowsA*columnsA*sizeof(double)));
    // CUDACHECK(cudaEventRecord(stopMalloc1[i]));
    // CUDACHECK(cudaEventRecord(startMemSet1[i],gpusInfo[i]->streams[0]));
    // CUDACHECK(cudaMemsetAsync(gpusInfo[i]->matrixDeviceA, 0, sizeof(double)*rowsA*columnsA,gpusInfo[i]->streams[0]));
    // CUDACHECK(cudaEventRecord(stopMemSet1[i],gpusInfo[i]->streams[0]));

}

template <class Toperation>
int MatrixUtilitiesCuda<Toperation>::getRealGpuId(int gpuRankOperation,int gpuSizeSystem)
{
    return gpuRankOperation%gpuSizeSystem;
}

template <class Toperation>
void MatrixUtilitiesCuda<Toperation>::cudaPrintMatrixCall(int rows,int columns,double* matrix)
{
    cublasHandle_t handle;
    CUBLASCHECK(cublasCreate(&handle));
    printf("HOLA\n");
    double* prueba;
    CUDACHECK(cudaMalloc ((void**)&prueba, rows*columns*sizeof(double)));
    CUDACHECK(cudaMemcpy(prueba,matrix,rows*columns*sizeof(double),cudaMemcpyHostToDevice));
    cudaPrintMatrix<<<1,1,1>>>(rows,columns,prueba);
    CUDACHECK(cudaDeviceSynchronize());
}

template class MatrixUtilitiesCuda<double>;