
/**
 * @brief Comprobador de errores de llamadas a librerías de Nvidia. 
 */
#pragma once
#include <cublas_v2.h>
#include <curand.h>
#include "nccl.h"

#ifdef CUBLAS_API_H_
/**
 * @brief Devuelve un string con el error de cublas en caso de que lo haya.
 * 
 * @param error , error de cublas.
 * @return const char* 
 */
static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}
#endif
/**
 * @brief Comprobar si hay errores de cuda.
 * 
 */
#define CUDACHECK(cmd)                                         \
	do                                                         \
	{                                                          \
		cudaError_t e = cmd;                                   \
		if (e != cudaSuccess)                                  \
		{                                                      \
			printf("Failed: Cuda error %s:%d '%s'\n",          \
				   __FILE__, __LINE__, cudaGetErrorString(e)); \
			exit(EXIT_FAILURE);                                \
		}                                                      \
	} while (0)

/**
 * @brief Comprobar si hay errores de cublas.
 * 
 */
#define CUBLASCHECK(cmd)                                   \
do                                                         \
{                                                          \
	cublasStatus_t s = cmd;                                \
	if (s != CUBLAS_STATUS_SUCCESS)                        \
	{                                                      \
		printf("Failed: Cublas error %s:%d '%s'\n",        \
				__FILE__, __LINE__, _cudaGetErrorEnum(s)); \
		exit(EXIT_FAILURE);                                \
	}                                                      \
} while (0)

/**
 * @brief Comprobar si hay errores de nccl.
 * 
 */
#define NCCLCHECK(cmd)                                         \
	do                                                         \
	{                                                          \
		ncclResult_t r = cmd;                                  \
		if (r != ncclSuccess)                                  \
		{                                                      \
			printf("Failed, NCCL error %s:%d '%s'\n",          \
				   __FILE__, __LINE__, ncclGetErrorString(r)); \
			exit(EXIT_FAILURE);                                \
		}                                                      \
    } while (0)

/**
 * @brief Comprobar error con curand
 * 
 */
#define CURAND_CALL(cmd) 										\
	do 															\
	{ 															\
		curandStatus_t x =cmd;									\
		if((x)!=CURAND_STATUS_SUCCESS) 							\
		{ 														\
    		printf("Error at %s:%d\n",__FILE__,__LINE__);		\
    		exit(EXIT_FAILURE);  								\
		}														\
	}while(0)

