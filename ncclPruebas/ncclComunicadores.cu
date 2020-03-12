#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

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

	struct GpuProperties
	{
		int nDevicesGlobal;
		int nDevicesOperation;
		int rankGlobal;
		int rankOperation;
		int rankRow;
		int rankCol;
		int *devicesGlobal;
		int *devicesOperation;
		int *devicesRow;
		int *devicesCol;
		ncclComm_t *commGlobal;
		ncclComm_t *commOperation;
		ncclComm_t *commRow;
		ncclComm_t *commCol;
		cudaStream_t *stream;
	
		GpuProperties(int devicesTotal,int rankGlobal)
		{
			this->nDevicesGlobal=devicesTotal;
			this->rankGlobal=rankGlobal;
		}
	
	};

__global__ void
cudaHelloWorld()
{
	printf("Hola\n");
}

int main(int argc, char *argv[])
{
	//Configuracion del comunicador que tiene a todos los dispositivos
	int nDev;
	max(cudaGetDeviceCount(&nDev), 4);
	GpuProperties xdd=GpuProperties(3,1);
	std::cout<<xdd.nDevicesGlobal<<std::endl;

	ncclComm_t comms[nDev];
	int devs[nDev];
	for (int i = 0; i < nDev; i++)
	{
		devs[i] = i;
	}
	cudaStream_t *s = (cudaStream_t *)malloc(sizeof(cudaStream_t) * nDev);

	//initializing NCCL
	NCCLCHECK(ncclCommInitAll(comms, nDev, devs));

	//finalizing NCCL
	for (int i = 0; i < nDev; ++i)
		ncclCommDestroy(comms[i]);

	std::cout << "Fin del programa" << std::endl;
	return 0;
}