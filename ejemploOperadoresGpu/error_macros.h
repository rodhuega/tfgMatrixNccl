
#ifndef CUDA_ERRORS
#define CUDA_ERRORS

#ifdef NODEFINED
#define CUDA_SAFE_CALL( call ) {                                         \
 cudaError_t err = call;                                                 \
 if( cudaSuccess != err ) {                                              \
   fprintf(stderr,"CUDA: error occurred in cuda routine. Exiting...\n"); \
   exit(err);                                                            \
 } }
 } }
#endif

#define CUBLAS_SAFE_CALL( call ) {                                         \
 cublasStatus_t err = call;                                                \
 if( CUBLAS_STATUS_SUCCESS != err ) {                                      \
   fprintf(stderr,"CUBLAS: error occurred in cuda routine. Exiting...\n"); \
   cublasDestroy(handle);                                                  \
   exit(err);                                                              \
 } }

#define CU_SAFE_CALL( call ) {                                         \
 CUresult err = call;                                                 \
 if( CUDA_SUCCESS != err ) {                                              \
   fprintf(stderr,"CUDA: error occurred in cuda routine. Exiting...\n"); \
   exit(err);                                                            \
 } }

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#define CUDA_SAFE_CALL( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

#endif

