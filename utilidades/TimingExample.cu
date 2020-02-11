
/************************************************
 * Simple CUDA example to transfer data CPU-GPU *
 ************************************************/

#include <stdio.h>
#include <stdlib.h>

#define CUDA_SAFE_CALL( call ) {                                         \
 cudaError_t err = call;                                                 \
 if( cudaSuccess != err ) {                                              \
   fprintf(stderr,"CUDA: error occurred in cuda routine. Exiting...\n"); \
   exit(err);                                                            \
 } }

#define	A(i,j)		A[ (j) + ((i)*(n)) ]
#define	B(i,j) 		B[ (j) + ((i)*(n)) ]

int main( int argc, char *argv[] ) {
  unsigned int m, n;
  unsigned int i, j;

  cudaEvent_t start, stop;
  float elapsedTime;

  /* Generating input data */
  if( argc<3 ) {
    printf("Usage: %s rows cols \n",argv[0]);
    exit(-1);
  }
  sscanf(argv[1],"%d",&m);
  sscanf(argv[2],"%d",&n);

  /* STEP 1: Allocate memory for three m-by-n matrices called A and B in the host */
  float *A = (float *) malloc( m*n*sizeof(float) );
  float *B = (float *) malloc( m*n*sizeof(float) );
  float *C = (float *) malloc( m*n*sizeof(float) );

  /* STEP 2: Fill matrices A and B with real values between -1.0 and 1.0 */
  printf("%s: Generating two random matrices of size %dx%d...\n",argv[0],m,n);
  for( i=0; i<m; i++ ) {
    for( j=0; j<n; j++ ) {
      A( i, j ) = 2.0f * ( (float) rand() / RAND_MAX ) - 1.0f;
      B( i, j ) = 2.0f * ( (float) rand() / RAND_MAX ) - 1.0f;
    }
  }
  printf("%s: Transferring data...\n",argv[0]);

  cudaEventCreate(&start);
  cudaEventRecord(start,0);

  /* STEP 3: Allocate memory for three m-by-n matrices called A, B, and C into the device memory */
  unsigned int mem_size = m * n * sizeof(float);
  float *d_A, *d_B, *d_C;
  CUDA_SAFE_CALL( cudaMalloc((void **) &d_A, mem_size ) );
  CUDA_SAFE_CALL( cudaMalloc((void **) &d_B, mem_size ) );
  CUDA_SAFE_CALL( cudaMalloc((void **) &d_C, mem_size ) );

  /* STEP 4: Copy host memory (only matrices A and B) to the device memory (matrices d_A and d_B) */
  CUDA_SAFE_CALL( cudaMemcpy( d_A, A, mem_size, cudaMemcpyHostToDevice ) );
  CUDA_SAFE_CALL( cudaMemcpy( d_B, B, mem_size, cudaMemcpyHostToDevice ) );

  /* STEP 5: Copy back from device memory into the host memory only data corresponding to matrix C (d_C) */
  CUDA_SAFE_CALL( cudaMemcpy( C, d_C, mem_size, cudaMemcpyDeviceToHost ) );

  cudaEventCreate(&stop);
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&elapsedTime, start,stop);
  printf("Elapsed time : %f ms\n" ,elapsedTime);

  /* STEP 6: Deallocate device memory */
  CUDA_SAFE_CALL( cudaFree(d_A) );
  CUDA_SAFE_CALL( cudaFree(d_B) );
  CUDA_SAFE_CALL( cudaFree(d_C) );

  /* STEP 7: Deallocate host memory */
  free(A);
  free(B);
  free(C);
  
}

