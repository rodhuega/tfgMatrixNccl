#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define N 12

void print_matrix(int nr_rows_A, int nr_cols_A,float* A) {

    for(int i = 0; i < nr_rows_A; ++i){
        for(int j = 0; j < nr_cols_A; ++j){
            std::cout << A[j * nr_rows_A + i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {
	
	int sizeMalloc=N*N* sizeof(float);
	float h_A[N][N]={{147,  67,  56, 124, 151, 111,  89,  24,  39, 141, 151,  52},
	{  8, 146,  74, 101, 112, 150,   9, 174,  67,   3,  48, 102},
	{ 87, 146,  86,  17, 198,  98,  43, 143, 155,  26, 160,  62},
	{187, 115, 104,  24, 152, 118, 138, 168, 193,  12,  61,  87},
	{ 12,  24, 167, 182, 145,  80,  54, 187, 177, 126,  55, 176},
	{ 65,   9,  60, 142,  18,  35, 130, 102, 177,  98, 161, 100},
	{ 24,  83, 178,  37, 195, 110, 140, 131, 158,  90, 141,  66},
	{174, 172,  81, 102,  27,  48, 138,  99, 110,  27,  20,  44},
	{ 89, 163, 150,  37,  27, 166, 120, 140,  36, 185,  63,  81},
	{177, 125, 179,  36,  79,  90, 195, 161, 119, 165,  29, 120},
	{154,   2, 151, 164, 174, 118,  25,  82, 110, 112, 138,  11},
	{152,  78, 111, 132, 176, 181,  34, 133, 155,  14,  88,   9}};
	float h_B[N][N]={{147,  67,  56, 124, 151, 111,  89,  24,  39, 141, 151,  52},
	{  8, 146,  74, 101, 112, 150,   9, 174,  67,   3,  48, 102},
	{ 87, 146,  86,  17, 198,  98,  43, 143, 155,  26, 160,  62},
	{187, 115, 104,  24, 152, 118, 138, 168, 193,  12,  61,  87},
	{ 12,  24, 167, 182, 145,  80,  54, 187, 177, 126,  55, 176},
	{ 65,   9,  60, 142,  18,  35, 130, 102, 177,  98, 161, 100},
	{ 24,  83, 178,  37, 195, 110, 140, 131, 158,  90, 141,  66},
	{174, 172,  81, 102,  27,  48, 138,  99, 110,  27,  20,  44},
	{ 89, 163, 150,  37,  27, 166, 120, 140,  36, 185,  63,  81},
	{177, 125, 179,  36,  79,  90, 195, 161, 119, 165,  29, 120},
	{154,   2, 151, 164, 174, 118,  25,  82, 110, 112, 138,  11},
	{152,  78, 111, 132, 176, 181,  34, 133, 155,  14,  88,   9}};
	float *h_C = (float *)malloc(sizeMalloc);

	float *d_A, *d_B, *d_C;
	cudaMalloc(&d_A,sizeMalloc);
	cudaMalloc(&d_B,sizeMalloc);
	cudaMalloc(&d_C,sizeMalloc);

	cudaMemcpy(d_A,h_A,sizeMalloc,cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,h_B,sizeMalloc,cudaMemcpyHostToDevice);

	std::cout << "A =" << std::endl;
	print_matrix(N, N,(float*) h_A);
	std::cout << "B =" << std::endl;
	print_matrix(N, N,(float*)h_B);

	float alfa = 1;
	float beta = 0;

	cublasHandle_t handle;
	cublasCreate(&handle);

	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alfa, d_A, N, d_B, N, &beta, d_C, N);

	cublasDestroy(handle);

	cudaMemcpy(h_C,d_C,sizeMalloc,cudaMemcpyDeviceToHost);
	std::cout << "C =" << std::endl;
	print_matrix( N, N,h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}