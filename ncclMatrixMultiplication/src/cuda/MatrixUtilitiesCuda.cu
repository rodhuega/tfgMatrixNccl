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


int MatrixUtilitiesCuda::matrixCalculateIndex(int rowSize, int columnSize, int rowIndex, int columnIndex)
{
    return IDX2C(rowIndex,columnIndex,rowSize);
    // return columnSize * rowIndex + columnIndex;
}

void MatrixUtilitiesCuda::cudaPrintMatrixCall(int rows,int columns,double* matrix)
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