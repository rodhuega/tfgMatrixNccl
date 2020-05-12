#include <stdio.h>
#include <unistd.h>
#include "cuda_runtime.h"
#include "nccl.h"

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

ncclComm_t* createNccl(int nDevices)
{
	ncclComm_t *comms= new ncclComm_t[nDevices];
	int *devs = new int[nDevices];

	for(int i =0;i<nDevices;i++)
	{
		devs[i]=i;
	}
	NCCLCHECK(ncclCommInitAll(comms, nDevices, devs));
	return comms;
}


int main(int argc, char* argv[])
{
	printf("Empieza \n");
	int nDevicesGlobal;
	CUDACHECK(cudaGetDeviceCount(&nDevicesGlobal));
	for(int kk=0;kk<30;kk++)
	{
		sleep(2);

		//managing 4 devices
		int nDev = nDevicesGlobal;
		int size = 32*1024*1024;
		
		ncclComm_t *comms=createNccl(nDevicesGlobal);

		//allocating and initializing device buffers
		float** sendbuff = (float**)malloc(nDev * sizeof(float*));
		float** recvbuff = (float**)malloc(nDev * sizeof(float*));
		cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);


		for (int i = 0; i < nDev; ++i) {
			CUDACHECK(cudaSetDevice(i));
			CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
			CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
			CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
			CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
			CUDACHECK(cudaStreamCreate(s+i));
		}


		//initializing NCCL


		//calling NCCL communication API. Group API is required when using
		//multiple devices per thread
		NCCLCHECK(ncclGroupStart());
		for (int i = 0; i < nDev; ++i)
			NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
			comms[i], s[i]));
		NCCLCHECK(ncclGroupEnd());


		//synchronizing on CUDA streams to wait for completion of NCCL operation
		for (int i = 0; i < nDev; ++i) {
		CUDACHECK(cudaSetDevice(i));
		CUDACHECK(cudaStreamSynchronize(s[i]));
		}


		//free device buffers
		for (int i = 0; i < nDev; ++i) {
		CUDACHECK(cudaSetDevice(i));
		CUDACHECK(cudaFree(sendbuff[i]));
		CUDACHECK(cudaFree(recvbuff[i]));
		CUDACHECK(cudaStreamDestroy(s[i]));

		}


		//finalizing NCCL
		for(int i = 0; i < nDev; ++i)
		{
			ncclCommDestroy(comms[i]);
		}


		printf("Success; %d\n",kk);
	}
	sleep(2);
	return 0;
}