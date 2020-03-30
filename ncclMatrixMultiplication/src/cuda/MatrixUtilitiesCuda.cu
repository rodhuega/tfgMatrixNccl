#include "MatrixUtilitiesCuda.cuh"


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


void MatrixUtilitiesCuda::cudaPrintMatrixCall(int rows,int columns,double* matrix)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    printf("HOLA\n");
    double* prueba;
    cudaMalloc ((void**)&prueba, rows*columns*sizeof(double));
    cudaMemcpy(prueba,matrix,rows*columns*sizeof(double),cudaMemcpyHostToDevice);
    cudaPrintMatrix<<<1,1,1>>>(rows,columns,prueba);
    cudaDeviceSynchronize();
}